# Rate Limit Configuration Guide

## Overview

Phase 25 adds configurable rate limit settings to optimize parallel chunk processing for different API provider tiers. This allows users to maximize performance while respecting their specific rate limits.

## Configuration Settings

Add these settings to your `config.yaml`:

```yaml
# ============================================================
# Rate Limit Settings (for chunk parallel processing)
# ============================================================

# OpenAI Rate Limits:
# - gpt-4o: 30,000 TPM / 500 RPM (Tokens/Requests Per Minute)
# - gpt-4o-mini: 500,000 TPM / 500 RPM
# Recommendation: Use 1-2 workers for gpt-4o, 3-4 for gpt-4o-mini
max_workers_openai: 2

# Gemini Rate Limits:
# - Free tier: 15 RPM (requests per minute)
# - Pay-as-you-go: Higher limits, varies by model
# Recommendation: Use 4-6 workers for paid tier, 2 for free tier
max_workers_gemini: 4

# How long to wait when hitting rate limits (seconds)
# OpenAI typically suggests waiting 30-60s after 429 error
rate_limit_retry_delay: 40.0

# Maximum retry attempts for rate-limited requests
# After this many retries, the chunk will be skipped
rate_limit_max_retries: 3
```

## Checking Your Rate Limits

### OpenAI
Visit https://platform.openai.com/settings/organization/limits

Example limits by model:
- **gpt-4o**: 30,000 TPM / 500 RPM
- **gpt-4o-mini**: 500,000 TPM / 500 RPM (16x more tokens!)
- **gpt-4o-realtime-preview**: 40,000 TPM / 200 RPM

### Gemini
Visit https://ai.google.dev/pricing

- **Free tier**: 15 RPM, 1 million TPM
- **Paid tier**: 2000 RPM, varies by model

## Optimization Guide

### For OpenAI

**Using gpt-4o (30K TPM):**
```yaml
max_workers_openai: 2  # Conservative, avoids rate limits
```

**Using gpt-4o-mini (500K TPM):**
```yaml
max_workers_openai: 4  # Can handle more parallel workers
```

**Free tier users:**
```yaml
max_workers_openai: 1  # Very conservative
rate_limit_retry_delay: 60.0  # Longer wait on errors
```

### For Gemini

**Paid tier:**
```yaml
max_workers_gemini: 6  # Can handle high concurrency
rate_limit_retry_delay: 30.0  # Shorter wait times
```

**Free tier:**
```yaml
max_workers_gemini: 2  # Respect 15 RPM limit
rate_limit_retry_delay: 45.0  # Be conservative
```

## How It Works

1. **Automatic Backend Detection**: System detects if you're using OpenAI or Gemini
2. **Worker Adjustment**: Uses `max_workers_openai` or `max_workers_gemini` accordingly
3. **Rate Limit Detection**: Catches 429 HTTP errors or "rate_limit" in error message
4. **Automatic Retry**: Waits `rate_limit_retry_delay` seconds and retries
5. **Max Retry Limit**: After `rate_limit_max_retries` attempts, skips the chunk
6. **Continued Processing**: Other chunks continue processing in parallel

## Example Scenario

**2.4-hour video (44 chunks) with gpt-4o:**

**Before Configuration (4 workers):**
```
✓ Chunk 1-32: Success
✗ Chunk 33: Rate limit error (30K TPM exceeded)
⏳ Waiting 40s...
✓ Chunk 33: Retry successful
✓ Chunk 34-44: Success
Total time: ~6 minutes (includes 40s wait)
```

**After Configuration (2 workers):**
```
✓ Chunk 1-44: All successful (no rate limits hit)
Total time: ~5 minutes (no waiting, just processing)
```

## Troubleshooting

### Still hitting rate limits?

**Solution 1: Reduce workers**
```yaml
max_workers_openai: 1  # Most conservative setting
```

**Solution 2: Increase retry delay**
```yaml
rate_limit_retry_delay: 60.0  # Wait longer between retries
```

**Solution 3: Check your API usage**
- Multiple applications using same API key?
- Processing multiple videos simultaneously?
- Consider upgrading API tier

### Processing too slow?

**Solution 1: Increase workers (if you have high tier)**
```yaml
max_workers_openai: 4  # For gpt-4o-mini (500K TPM)
max_workers_gemini: 6  # For Gemini paid tier
```

**Solution 2: Use a faster model**
- Switch from `gpt-4o` to `gpt-4o-mini` (16x higher TPM limit)
- Switch from `gemini-1.5-pro` to `gemini-1.5-flash` (cheaper, faster)

### Chunks failing after retries?

Check logs for:
- Network connectivity issues
- API key validity
- Account billing status
- Service outages

## Implementation Details

**Files Modified:**
- `src/config_manager.py` - AppConfig dataclass with rate limit fields
- `src/vision_backends.py` - Uses config values instead of hardcoded
- `app.py` - Passes config to analyze_video_frames()
- `config.yaml` - Added rate limit settings
- `config.yaml.example` - Documentation and examples
- `README.md` - User-facing configuration guide

**Backward Compatibility:**
- If config values not found, uses safe defaults (2/4/40/3)
- Existing config files work without modification
- All 38 tests passing ✅

## Benefits

✅ **Optimized for your API tier** - Configure based on your specific limits  
✅ **Avoid rate limit errors** - Conservative defaults prevent 429 errors  
✅ **Maximize performance** - Higher tier users can use more workers  
✅ **Automatic retry** - Handles transient rate limits gracefully  
✅ **Production-ready** - Enterprise-grade configuration management  
✅ **Fully documented** - Clear guidance for all API tiers  

## Next Steps

1. Check your API provider dashboard for current limits
2. Update `config.yaml` with optimal values for your tier
3. Test with a long video to verify performance
4. Monitor logs for any rate limit messages
5. Adjust settings as needed based on results

For questions or issues, refer to:
- `README.md` - User guide with examples
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- Provider documentation linked above
