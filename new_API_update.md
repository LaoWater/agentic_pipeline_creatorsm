We received an update from our Back-end&Front-end team.

The API pipeline now comes with extra fields - we will need to integrate this in the whole current ecosystem.

New API Payload Structure
The frontend now sends a ContentGeneratorData object with this enhanced structure:


{
  "company": {
    "id": "uuid",
    "name": "string",
    "mission": "string", 
    "tone_of_voice": "string",
    "primary_color_1": "#hex",
    "primary_color_2": "#hex",
    "logo_path": "string|null"
  },
  "content": {
    "topic": "string",
    "description": "string", 
    "hashtags": ["string[]"],
    "call_to_action": "string"
  },
  "image_control": {
    "level_1": {
      "enabled": boolean,
      "style": "string",
      "guidance": "string", 
      "caption": "string",
      "ratio": "string",
      "starting_image_url": "string|null"
    },
    "level_2": {
      "facebook": {
        "enabled": boolean,
        "style": "string",
        "guidance": "string",
        "caption": "string", 
        "ratio": "string",
        "starting_image_url": "string|null"
      },
      // Similar structure for instagram, linkedin, twitter
    }
  },
  "platforms": [
    {
      "platform": "facebook|instagram|linkedin|twitter",
      "post_type": "string",
      "selected": boolean
    }
  ]
}



Our current Pipeline is in "/agentic_pipeline_creatorsm_fullgemini".

---

Study "api_models.py" to understand old (working) pipeline in PipelineRequest.

Next, understand how the data is being used in pipeline_orchestrator, main, llm_services, visual_model.

In config.py we handle most of the LLM prompt engineering.


---- 


This update is a lot about image generation - we are now no longer allowing full pass of the LLM text - but we add to it - either when building the visual model prompt or even after the prompt is built and before feeding to visual model (appending the key Image Controls)
Image Control level 2 is Platform specific and over-rides level 1.


# Expected Behavior:
Level 2 always overrides Level 1 when both exist for a platform
Starting images are downloaded and used in visual generation
Style, guidance, and caption controls are properly injected into prompts
Ratio controls affect image generation parameters





## LOW DIAGRAM
Frontend → API → Pipeline Orchestrator → Platform Generator
                                      ↓
    Level 1 (Global) ←-----------→ Level 2 (Platform)
                                      ↓
                            Effective Image Control
                                      ↓
                            LLM Service (Enhanced Prompts)
                                      ↓
                            Visual Model (Enhanced Visuals)


                            