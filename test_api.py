"""
Test script for the Call Center Compliance API
Supports both sync and async modes
"""
import httpx
import time
import json
import argparse

BASE_URL = "http://localhost:8000"
TEST_AUDIO_URL = "https://recordings.exotel.com/exotelrecordings/guvi64/5780094ea05a75c867120809da9a199f.mp3"


def test_health():
    """Test health endpoint"""
    print("\n--- Testing Health Endpoint ---")
    response = httpx.get(f"{BASE_URL}/health")
    print(f"Health check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def test_api_info():
    """Test API info endpoint"""
    print("\n--- Testing API Info ---")
    response = httpx.get(f"{BASE_URL}/api")
    print(f"API info: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def test_providers():
    """Test providers endpoint"""
    print("\n--- Testing Providers ---")
    response = httpx.get(f"{BASE_URL}/providers")
    print(f"Providers: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def test_analyze_sync(audio_url: str = TEST_AUDIO_URL):
    """Test synchronous analysis (no Celery required)"""
    print(f"\n--- Testing Sync Analysis ---")
    print(f"Audio URL: {audio_url}")
    print("This may take 30-120 seconds...")
    
    try:
        # Use longer timeout for sync processing
        with httpx.Client(timeout=180.0) as client:
            response = client.post(
                f"{BASE_URL}/analyze/sync",
                json={
                    "audio_url": audio_url,
                    "sop_template": "standard",
                    "enable_pii_redaction": False
                }
            )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
        
        result = response.json()
        status = result.get("status")
        
        if status == "completed":
            print("\n" + "="*50)
            print("ANALYSIS COMPLETE!")
            print("="*50)
            
            # Print summary
            print(f"\nSummary: {result.get('summary', 'N/A')}")
            print(f"Processing time: {result.get('processing_time_seconds', 0):.1f}s")
            print(f"LLM Provider: {result.get('llm_provider_used', 'N/A')}")
            
            # Print SOP validation
            sop = result.get('sop_validation', {})
            print(f"\nSOP Compliance: {sop.get('overall_compliance', 0)}%")
            
            # Print keywords
            keywords = result.get('keywords', [])
            print(f"\nKeywords: {', '.join(keywords[:10]) if keywords else 'None'}")
            
            # Full JSON (truncated)
            full_json = json.dumps(result, indent=2, default=str)
            if len(full_json) > 2000:
                print(f"\nFull response (truncated):\n{full_json[:2000]}...")
            else:
                print(f"\nFull response:\n{full_json}")
            
            return True
        else:
            print(f"Task status: {status}")
            print(f"Error: {result.get('error', 'Unknown')}")
            return False
            
    except httpx.TimeoutException:
        print("Request timed out. The audio file may be too long or the server is overloaded.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_analyze_async(audio_url: str = TEST_AUDIO_URL):
    """Submit audio for analysis and poll for results (requires Celery)"""
    print(f"\n--- Testing Async Analysis ---")
    print(f"Audio URL: {audio_url}")
    
    # Submit analysis request
    response = httpx.post(
        f"{BASE_URL}/analyze",
        json={"audio_url": audio_url}
    )
    
    if response.status_code != 200:
        print(f"Error submitting task: {response.status_code}")
        print(response.text)
        return False
    
    result = response.json()
    task_id = result["task_id"]
    print(f"Task submitted successfully. Task ID: {task_id}")
    
    # Poll for results
    max_attempts = 60  # 5 minutes max
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        print(f"\nPolling attempt {attempt}/{max_attempts}...")
        
        poll_response = httpx.get(f"{BASE_URL}/result/{task_id}")
        poll_result = poll_response.json()
        
        status = poll_result.get("status")
        print(f"Status: {status}")
        
        if status == "completed":
            print("\n" + "="*50)
            print("ANALYSIS COMPLETE!")
            print("="*50)
            print(json.dumps(poll_result, indent=2, default=str))
            return True
        
        elif status == "failed":
            print(f"\nTask failed: {poll_result.get('error')}")
            return False
        
        time.sleep(5)  # Wait 5 seconds between polls
    
    print("Timeout waiting for results")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Call Center Compliance API")
    parser.add_argument("--mode", choices=["sync", "async", "health"], default="sync",
                       help="Test mode: sync (no Celery), async (requires Celery), or health only")
    parser.add_argument("--audio-url", default=TEST_AUDIO_URL, help="Audio URL to analyze")
    
    args = parser.parse_args()
    
    print("Call Center Compliance API - Test Script")
    print("="*50)
    
    # Always test health first
    if not test_health():
        print("Health check failed! Is the server running?")
        exit(1)
    
    test_api_info()
    test_providers()
    
    if args.mode == "health":
        print("\nHealth checks passed!")
        exit(0)
    
    # Test analysis based on mode
    if args.mode == "sync":
        if not test_analyze_sync(args.audio_url):
            print("\nSync analysis test failed!")
            exit(1)
    else:
        if not test_analyze_async(args.audio_url):
            print("\nAsync analysis test failed!")
            exit(1)
    
    print("\nAll tests passed!")
