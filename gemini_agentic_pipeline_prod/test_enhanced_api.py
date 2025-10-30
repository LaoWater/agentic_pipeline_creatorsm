# test_enhanced_api.py
import asyncio
import json
from api_models import (
    ContentGeneratorData, Company, Content, ImageControl, 
    ImageControlLevel1, ImageControlLevel2, PlatformImageControl, Platform
)
from pipeline_orchestrator import generate_enhanced_social_media_posts_pipeline

async def test_enhanced_pipeline():
    """Test the enhanced pipeline with the new ContentGeneratorData structure."""
    
    # Create test data matching the new API structure
    test_data = ContentGeneratorData(
        company=Company(
            id="test-company-123",
            name="Creators Multiverse",
            mission="Empowering creators to build their digital presence with AI-powered tools",
            tone_of_voice="Inspirational & Empowering with a Cosmic/Magical theme",
            primary_color_1="#6366f1",  # Purple
            primary_color_2="#ec4899",  # Pink
            logo_path=None
        ),
        content=Content(
            topic="Introducing our revolutionary AI content creation platform",
            description="Launch announcement for our new AI-powered social media content generation tool that helps creators build amazing content across platforms",
            hashtags=["AI", "ContentCreation", "SocialMedia", "CreatorsMultiverse"],
            call_to_action="Visit our website to start creating amazing content today!"
        ),
        image_control=ImageControl(
            level_1=ImageControlLevel1(
                enabled=True,
                style="Modern, vibrant, tech-inspired digital art",
                guidance="Focus on creativity and innovation, use cosmic/magical elements subtly",
                caption="AI-powered content creation revolution",
                ratio="1:1",
                starting_image_url=None
            ),
            level_2=ImageControlLevel2(
                instagram=PlatformImageControl(
                    enabled=True,
                    style="Instagram-optimized, colorful and engaging",
                    guidance="Perfect for Instagram feed, make it pop with vibrant colors",
                    caption="Perfect for your Instagram feed",
                    ratio="1:1",
                    starting_image_url=None
                ),
                linkedin=PlatformImageControl(
                    enabled=True,
                    style="Professional, clean, business-oriented",
                    guidance="Corporate and professional look suitable for LinkedIn",
                    caption="Professional announcement",
                    ratio="16:9",  # Valid Google Imagen ratio
                    starting_image_url=None
                )
            )
        ),
        platforms=[
            Platform(platform="instagram", post_type="Image", selected=True),
            Platform(platform="linkedin", post_type="Image", selected=True),
            Platform(platform="twitter", post_type="Text", selected=True),
            Platform(platform="facebook", post_type="Image", selected=False)
        ],
        language="English",
        upload_to_cloud=False  # Set to False for testing to avoid cloud operations
    )
    
    print("🧪 Testing Enhanced Pipeline with new ContentGeneratorData structure")
    print("=" * 70)
    
    # Print test configuration
    print(f"Company: {test_data.company.name}")
    print(f"Topic: {test_data.content.topic}")
    print(f"Selected Platforms: {[p.platform for p in test_data.platforms if p.selected]}")
    print(f"Image Control Level 1 Enabled: {test_data.image_control.level_1.enabled}")
    
    level_2_overrides = []
    if test_data.image_control.level_2.instagram and test_data.image_control.level_2.instagram.enabled:
        level_2_overrides.append("instagram")
    if test_data.image_control.level_2.linkedin and test_data.image_control.level_2.linkedin.enabled:
        level_2_overrides.append("linkedin")
    
    print(f"Level 2 Overrides: {level_2_overrides}")
    print("=" * 70)
    
    try:
        # Run the enhanced pipeline
        result = await generate_enhanced_social_media_posts_pipeline(test_data)
        
        print("\n✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Print results summary
        print(f"Pipeline ID: {result.get('pipeline_id')}")
        print(f"Posts Generated: {len(result.get('posts', []))}")
        
        for post in result.get('posts', []):
            print(f"\n📱 {post['platform'].upper()} ({post['post_type']}):")
            print(f"   Text: {post['original_text_content'][:100]}...")
            if post.get('media_asset'):
                print(f"   Media: {post['media_asset']['file_path']}")
            if post.get('media_generation_prompt_used'):
                print(f"   Enhanced Prompt: {post['media_generation_prompt_used'][:80]}...")
        
        print(f"\nImage Controls Summary:")
        controls_used = result.get('image_controls_used', {})
        print(f"   Level 1 Enabled: {controls_used.get('level_1_enabled')}")
        print(f"   Level 2 Overrides: {controls_used.get('level_2_overrides')}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_api_models():
    """Test that the new API models work correctly."""
    print("\n🧪 Testing API Model Validation")
    print("=" * 50)
    
    try:
        # Test creating a complete ContentGeneratorData object
        test_data = {
            "company": {
                "id": "test-123",
                "name": "Test Company",
                "mission": "Test mission",
                "tone_of_voice": "Professional",
                "primary_color_1": "#000000",
                "primary_color_2": "#ffffff",
                "logo_path": None
            },
            "content": {
                "topic": "Test topic",
                "description": "Test description",
                "hashtags": ["test", "example"],
                "call_to_action": "Test CTA"
            },
            "image_control": {
                "level_1": {
                    "enabled": True,
                    "style": "test style",
                    "guidance": "test guidance",
                    "caption": "test caption",
                    "ratio": "1:1",
                    "starting_image_url": None
                },
                "level_2": {
                    "instagram": {
                        "enabled": True,
                        "style": "instagram style",
                        "guidance": "instagram guidance",
                        "caption": "instagram caption",
                        "ratio": "1:1",
                        "starting_image_url": None
                    }
                }
            },
            "platforms": [
                {
                    "platform": "instagram",
                    "post_type": "Image",
                    "selected": True
                }
            ]
        }
        
        # Validate with Pydantic
        validated_data = ContentGeneratorData(**test_data)
        print("✅ API Model validation successful!")
        print(f"   Company: {validated_data.company.name}")
        print(f"   Platforms: {len(validated_data.platforms)}")
        print(f"   Image Control Level 1: {validated_data.image_control.level_1.enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Model validation failed: {str(e)}")
        return False

async def main():
    """Main test function."""
    print("🚀 ENHANCED API INTEGRATION TESTS")
    print("=" * 70)
    
    # Test 1: API Models
    model_test_passed = test_api_models()
    
    # Test 2: Enhanced Pipeline (only if models pass)
    if model_test_passed:
        pipeline_test_passed = await test_enhanced_pipeline()
    else:
        pipeline_test_passed = False
    
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"API Models Test: {'✅ PASSED' if model_test_passed else '❌ FAILED'}")
    print(f"Enhanced Pipeline Test: {'✅ PASSED' if pipeline_test_passed else '❌ FAILED'}")
    
    if model_test_passed and pipeline_test_passed:
        print("\n🎉 ALL TESTS PASSED! Enhanced API integration is ready.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())