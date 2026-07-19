from pydantic import networks
import json
import asyncio
import httpx
import logging
from typing import Dict, List, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END,START
from pydantic import BaseModel, Field

import google.generativeai as genai
from config import settings
from services.search import web_search
from services.prompts import get_prompt_resource

logger = logging.getLogger(__name__)

from langfuse import get_client
langfuse = get_client()


# Configure Gemini
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)

# Define State Schema
class AgentState(TypedDict):
    video_path: str
    prompt_hint: Optional[str]
    music_vibe: str
    model_name: str
    search_queries: List[str]
    search_results: Dict[str, str]
    script_segments: List[Dict[str, Any]]
    error: Optional[str]

# Structured output Pydantic models
class LandmarkSearch(BaseModel):
    queries: List[str] = Field(description="List of search engine queries to find historical details about landmarks seen in the video.")
    visual_summary: str = Field(description="A brief visual summary of the video clips.")

class VlogSegmentSchema(BaseModel):
    start_second: int = Field(description="Timestamp in seconds where segment begins.")
    end_second: int = Field(description="Timestamp in seconds where segment ends.")
    voiceover_text: str = Field(description="The engaging, informative first-person vlog narration overlay.")
    music_vibe: str = Field(description="Music vibe match for this segment.")

class CompleteScriptSchema(BaseModel):
    segments: List[VlogSegmentSchema]

# Nodes
async def analyze_video_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: analyze_video")
    video_path = state["video_path"]
    model_name = state.get("model_name", settings.DEFAULT_ANALYSIS_MODEL)
    
    import os
    from services.video_description import get_video_description, save_video_description
    
    file_name = os.path.basename(video_path)
    
    # Extract filename_prefix
    if "--v_" in file_name:
        filename_prefix = file_name.split("--v_")[0]
    else:
        filename_prefix = os.path.splitext(file_name)[0]
        
    cached_desc = get_video_description(filename_prefix)
    if cached_desc:
        logger.info(f"Using cached description for {filename_prefix}")
        with langfuse.start_as_current_observation(
            as_type="span",
            name="cache_hit_video_description",
            input={"filename_prefix": filename_prefix},
            output=cached_desc
        ):
            pass
        return {
            "search_queries": cached_desc.get("queries", [])
        }
        
    # Ollama text-only models require Gemini or Mock fallback for video analysis
    is_ollama = model_name.startswith("ollama/")
    
    if not settings.GEMINI_API_KEY:
        logger.warning("No GEMINI_API_KEY found. Falling back to mock video analysis...")
        return {
            "search_queries": ["Rohtang Pass history", "Rohtang Pass motorcycling route"],
            "error": "Using mock analysis due to missing API key."
        }
        
    if is_ollama:
        logger.info(f"Selected model is local Ollama ({model_name}). Using Gemini for visual analysis step...")
        
    try:
        # Upload file to Gemini File API
        logger.info(f"Uploading video {video_path} to Gemini...")
        video_file = await asyncio.to_thread(genai.upload_file, path=video_path)
        
        # Poll file processing status
        while video_file.state.name == "PROCESSING":
            logger.info("Video is processing in Gemini Cloud...")
            await asyncio.sleep(5)
            video_file = await asyncio.to_thread(genai.get_file, video_file.name)
            
        if video_file.state.name == "FAILED":
            raise Exception("Gemini video ingestion processing failed.")
            
        logger.info("Ingested video successfully. Prompting Gemini for visual parsing...")
        
        # Use gemini-1.5-pro or DEFAULT_ANALYSIS_MODEL for structured response
        model = genai.GenerativeModel(model_name if "gemini" in model_name else settings.DEFAULT_ANALYSIS_MODEL)
        
        prompt = get_prompt_resource("video_analysis")
        
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="gemini_video_analysis",
            model=model.model_name,
            input={"prompt": prompt}
        ) as generation:
            response = await asyncio.to_thread(
                model.generate_content,
                contents=[video_file, prompt],
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": LandmarkSearch
                }
            )
            generation.update(output=response.text)
        
        # Delete file after parsing
        await asyncio.to_thread(genai.delete_file, video_file.name)
        
        parsed = json.loads(response.text)
        save_video_description(filename_prefix, response.text)
        return {
            "search_queries": parsed.get("queries", [])
        }
        
    except Exception as e:
        logger.error(f"analyze_video error: {str(e)}", exc_info=True)
        # Graceful fallback to mock queries
        return {
            "search_queries": ["Beautiful mountain roads", "Scenic driving routes"],
            "error": f"Analysis failed: {str(e)}"
        }

async def research_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: research")
    queries = state.get("search_queries", [])
    results = {}
    
    # Run searches concurrently
    tasks = [web_search(q) for q in queries]
    search_outputs = await asyncio.gather(*tasks)
    
    for q, out in zip(queries, search_outputs):
        results[q] = out
        
    return {
        "search_results": results
    }

async def generate_script_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Node: generate_script")
    video_path = state["video_path"]
    prompt_hint = state.get("prompt_hint", "")
    music_vibe = state.get("music_vibe", "Cinematic & Adventurous")
    search_results = state.get("search_results", {})
    model_name = state.get("model_name", settings.DEFAULT_RESEARCH_MODEL)
    
    # 1. Local Ollama Generation Logic
    if model_name.startswith("ollama/"):
        ollama_model = model_name.replace("ollama/", "")
        logger.info(f"Invoking local Ollama chat model: {ollama_model}...")
        
        research_context = ""
        for query, content in search_results.items():
            research_context += f"Query: {query}\nInformation:\n{content}\n\n"
            
        system_prompt = get_prompt_resource("ollama_system")
        
        user_prompt = get_prompt_resource("ollama_user").format(
            prompt_hint=prompt_hint,
            music_vibe=music_vibe,
            research_context=research_context
        )
        
        try:
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="ollama_script_generation",
                model=ollama_model,
                input={"system_prompt": system_prompt, "user_prompt": user_prompt}
            ) as generation:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{settings.OLLAMA_HOST}/api/chat",
                        json={
                            "model": ollama_model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "options": {
                                "temperature": 0.7
                            },
                            "stream": False,
                            "format": "json"
                        },
                        timeout=90.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        content = data.get("message", {}).get("content", "{}")
                        generation.update(output=content)
                        parsed = json.loads(content)
                        segments = parsed.get("segments", [])
                        
                        dict_segments = []
                        for s in segments:
                            dict_segments.append({
                                "start_second": int(s.get("start_second", 0)),
                                "end_second": int(s.get("end_second", 10)),
                                "voiceover_text": s.get("voiceover_text", ""),
                                "music_vibe": s.get("music_vibe", music_vibe)
                            })
                        return {
                            "script_segments": dict_segments
                        }
                    else:
                        raise Exception(f"Ollama server returned {response.status_code}")
        except Exception as e:
            logger.warning(f"Ollama execution failed: {str(e)}. Falling back to mock...")
            model_name = "mock" # trigger mock script generator below
            
    if not settings.GEMINI_API_KEY or model_name == "mock":
        logger.info("No GEMINI_API_KEY or model is mock. Generating mock script segments...")
        # Mock script segments based on video duration
        # We assume a 30 second mock video
        return {
            "script_segments": [
                {
                    "start_second": 0,
                    "end_second": 10,
                    "voiceover_text": f"Hey everyone! Today we are hitting the road. {prompt_hint or 'Enjoy this beautiful scenic view!'}",
                    "music_vibe": music_vibe
                },
                {
                    "start_second": 10,
                    "end_second": 20,
                    "voiceover_text": "The wind is in our faces and the scenery is absolutely breathtaking. This is what travel is all about.",
                    "music_vibe": music_vibe
                },
                {
                    "start_second": 20,
                    "end_second": 30,
                    "voiceover_text": "Thanks for tagging along! Don't forget to like and subscribe for more adventures.",
                    "music_vibe": music_vibe
                }
            ]
        }

    try:
        # Reconstruct query context from search results
        research_context = ""
        for query, content in search_results.items():
            research_context += f"Query: {query}\nInformation:\n{content}\n\n"
            
        model = genai.GenerativeModel(model_name if "gemini" in model_name else settings.DEFAULT_RESEARCH_MODEL)
        
        prompt = get_prompt_resource("gemini_script").format(
            prompt_hint=prompt_hint,
            music_vibe=music_vibe,
            research_context=research_context
        )
        
        # We need to estimate video duration, we can upload file again or since we already did, we pass a constraint.
        # Let's request structured json
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="gemini_script_generation",
            model=model.model_name,
            input={"prompt": prompt}
        ) as generation:
            response = await asyncio.to_thread(
                model.generate_content,
                contents=[prompt],
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": CompleteScriptSchema
                }
            )
            generation.update(output=response.text)
        
        parsed = json.loads(response.text)
        segments = parsed.get("segments", [])
        
        # Convert back to standard dicts
        dict_segments = []
        for s in segments:
            dict_segments.append({
                "start_second": s.get("start_second"),
                "end_second": s.get("end_second"),
                "voiceover_text": s.get("voiceover_text"),
                "music_vibe": s.get("music_vibe", music_vibe)
            })
            
        return {
            "script_segments": dict_segments
        }
        
    except Exception as e:
        logger.error(f"generate_script error: {str(e)}", exc_info=True)
        # Fallback script
        return {
            "script_segments": [
                {
                    "start_second": 0,
                    "end_second": 15,
                    "voiceover_text": f"Cruising through some incredible locations. {prompt_hint}",
                    "music_vibe": music_vibe
                }
            ],
            "error": f"Script gen failed: {str(e)}"
        }

# Compile Workflow Graph
def build_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("analyze_video", analyze_video_node)
    workflow.add_node("research", research_node)
    workflow.add_node("generate_script", generate_script_node)
    
    # Add Edges
    workflow.add_edge(START, "analyze_video")
    workflow.add_edge("analyze_video", "research")
    workflow.add_edge("research", "generate_script")
    workflow.add_edge("generate_script", END)
    
    return workflow.compile()

# Thread-safe execution wrapper
async def run_vlog_agent(
    video_path: str, 
    prompt_hint: Optional[str] = None, 
    music_vibe: str = "Cinematic & Adventurous",
    model_name: Optional[str] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    model_name = model_name or settings.DEFAULT_ANALYSIS_MODEL
    app = build_agent_graph()
    initial_state = {
        "video_path": video_path,
        "prompt_hint": prompt_hint,
        "music_vibe": music_vibe,
        "model_name": model_name,
        "search_queries": [],
        "search_results": {},
        "script_segments": [],
        "error": None
    }
    
    # Configure Langfuse callback handler if keys are set
    config = {}
    if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
        try:
            from langfuse.langchain import CallbackHandler
            import os
            langfuse_handler = CallbackHandler(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
                trace_name=f"Vlog Director Pipeline - {os.path.basename(video_path)}"
            )
            config["callbacks"] = [langfuse_handler]
            config["metadata"] = {
                "langfuse_tags": ["vlog-director-backend"],
                "video_path": video_path,
                "model_name": model_name,
                "music_vibe": music_vibe
            }
            if job_id:
                config["metadata"]["langfuse_session_id"] = job_id
            logger.info("Langfuse callback handler initialized successfully for tracing.")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse callback handler: {str(e)}", exc_info=True)
            
    result = await app.ainvoke(initial_state, config=config)
    return result
