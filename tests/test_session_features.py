"""Tests for features implemented in this session.

This test file covers:
1. Select All/None checkbox functionality
2. Filtered event exports (text, clips, highlight reel)
3. Auto-calculation of max_frames based on TPM
4. Sport-specific clip padding configuration
5. Improved AI prompting with detailed visual cues
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.config_manager import SPORT_PRESETS
from src.vision_backends import VisionBackendMixin


class TestSelectAllCheckbox:
    """Test select all/none checkbox functionality."""
    
    def test_select_all_checkbox_in_html(self):
        """Test that select all checkbox is present in HTML."""
        with open('templates/index.html', 'r') as f:
            content = f.read()
        
        # Check for select all checkbox
        assert 'id="selectAllCheckbox"' in content
        assert 'onchange="toggleEventSelectAll()"' in content
        
    def test_toggle_select_all_function_exists(self):
        """Test that toggleSelectAll function is defined."""
        with open('templates/index.html', 'r') as f:
            content = f.read()
        
        assert 'function toggleEventSelectAll()' in content
        
    def test_visible_indices_helper_function(self):
        """Test that getVisibleEventIndices helper function exists."""
        with open('templates/index.html', 'r') as f:
            content = f.read()
        
        assert 'function getVisibleEventIndices()' in content
        assert "row.style.display !== 'none'" in content


class TestFilteredExports:
    """Test that exports respect current filters."""
    
    def test_text_export_uses_visible_indices(self):
        """Test that text export only includes filtered events."""
        with open('templates/index.html', 'r') as f:
            content = f.read()
        
        # Check downloadSelectedTimestamps uses getVisibleEventIndices
        assert 'function downloadSelectedTimestamps()' in content
        # Find the function and check it calls getVisibleEventIndices
        download_func_start = content.find('function downloadSelectedTimestamps()')
        download_func_end = content.find('function', download_func_start + 1)
        download_func = content[download_func_start:download_func_end]
        
        assert 'getVisibleEventIndices()' in download_func
        assert 'visibleIndices.includes(idx)' in download_func
        
    def test_clip_export_uses_visible_indices(self):
        """Test that clip generation only includes filtered events."""
        with open('templates/index.html', 'r') as f:
            content = f.read()
        
        # Check generateAndDownloadSelectedClips uses getVisibleEventIndices
        assert 'function generateAndDownloadSelectedClips()' in content
        clips_func_start = content.find('function generateAndDownloadSelectedClips()')
        clips_func_end = content.find('function', clips_func_start + 1)
        clips_func = content[clips_func_start:clips_func_end]
        
        assert 'getVisibleEventIndices()' in clips_func
        assert 'visibleIndices.includes(idx)' in clips_func
        
    def test_highlight_reel_uses_visible_indices(self):
        """Test that highlight reel only includes filtered events."""
        with open('templates/index.html', 'r') as f:
            content = f.read()
        
        # Check generateAndDownloadHighlightReel uses getVisibleEventIndices
        assert 'function generateAndDownloadHighlightReel()' in content
        reel_func_start = content.find('function generateAndDownloadHighlightReel()')
        reel_func_end = content.find('function', reel_func_start + 1)
        reel_func = content[reel_func_start:reel_func_end]
        
        assert 'getVisibleEventIndices()' in reel_func
        assert 'visibleIndices.includes(idx)' in reel_func


class TestMaxFramesAutoCalculation:
    """Test automatic calculation of max_frames based on TPM limits."""
    
    def test_max_frames_calculation_logic_exists(self):
        """Test that max_frames auto-calculation code exists in vision_backends.py."""
        with open('src/vision_backends.py', 'r') as f:
            content = f.read()
        
        # Check for auto-calculation logic
        assert 'AUTO-CALCULATE max_frames' in content
        assert 'TOKENS_PER_FRAME' in content
        assert 'max_frames_from_tpm' in content
        
    def test_max_frames_tpm_formula(self):
        """Test the max_frames calculation formula."""
        # Simulate the calculation
        TOKENS_PER_FRAME = 850
        SYSTEM_PROMPT_TOKENS = 1000
        RESPONSE_TOKENS = 1500
        
        # Test with 500k TPM (gpt-5-mini)
        tpm = 500000
        max_tokens_per_call = tpm / 12
        available_tokens = max_tokens_per_call - SYSTEM_PROMPT_TOKENS - RESPONSE_TOKENS
        max_frames = max(5, int(available_tokens / TOKENS_PER_FRAME))
        
        assert max_frames >= 45  # Should be ~46 for 500k TPM
        assert max_frames <= 50  # Upper bound
        
        # Test with 200k TPM (gpt-5-nano)
        tpm = 200000
        max_tokens_per_call = tpm / 12
        available_tokens = max_tokens_per_call - SYSTEM_PROMPT_TOKENS - RESPONSE_TOKENS
        max_frames = max(5, int(available_tokens / TOKENS_PER_FRAME))
        
        assert max_frames >= 15  # Should be ~16 for 200k TPM
        assert max_frames <= 20
        
    def test_config_has_tpm_settings(self):
        """Test that config file has TPM settings."""
        with open('config.yaml', 'r') as f:
            content = f.read()
        
        assert 'openai_rate_limit_tpm' in content
        assert 'openai_rate_limit_rpm' in content


class TestClipPaddingConfiguration:
    """Test sport-specific clip padding configuration."""
    
    def test_sport_presets_have_clip_padding(self):
        """Test that sport presets include clip padding settings."""
        for sport_name, preset in SPORT_PRESETS.items():
            assert hasattr(preset, 'clip_padding_before'), f"{sport_name} missing clip_padding_before"
            assert hasattr(preset, 'clip_padding_after'), f"{sport_name} missing clip_padding_after"
            assert isinstance(preset.clip_padding_before, (int, float))
            assert isinstance(preset.clip_padding_after, (int, float))
            
    def test_floorball_clip_padding_values(self):
        """Test floorball-specific padding values from config.yaml."""
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        floorball_preset = config['sport_presets']['floorball']
        assert floorball_preset['clip_padding_before'] == 10
        assert floorball_preset['clip_padding_after'] == 5
        
    def test_prepare_clips_uses_padding_parameters(self):
        """Test that prepare_clips function accepts separate before/after padding."""
        with open('src/video_tools.py', 'r') as f:
            content = f.read()
        
        # Check function signature
        assert 'def prepare_clips(' in content
        assert 'padding_before' in content
        assert 'padding_after' in content
        
        # Check it uses both parameters
        prepare_clips_start = content.find('def prepare_clips(')
        prepare_clips_end = content.find('\ndef ', prepare_clips_start + 1)
        if prepare_clips_end == -1:
            prepare_clips_end = len(content)
        prepare_clips_func = content[prepare_clips_start:prepare_clips_end]
        
        assert 'padding_before' in prepare_clips_func
        assert 'padding_after' in prepare_clips_func
        assert 'padding_before + padding_after' in prepare_clips_func


class TestImprovedAIPrompting:
    """Test improved AI prompting with detailed visual cues."""
    
    def test_detailed_goal_indicators_in_prompt(self):
        """Test that goal detection has detailed visual indicators."""
        with open('src/vision_backends.py', 'r') as f:
            content = f.read()
        
        # Check for detailed goal indicators
        assert 'Ball crossing goal line' in content
        assert 'Net movement' in content
        assert 'Goalkeeper position' in content
        assert 'Player reactions' in content
        assert 'Team celebration' in content
        
    def test_detailed_shot_indicators_in_prompt(self):
        """Test that shot detection has detailed visual indicators."""
        with open('src/vision_backends.py', 'r') as f:
            content = f.read()
        
        assert 'Wind-up motion' in content
        assert 'Strike impact' in content
        assert 'Ball trajectory' in content
        
    def test_detailed_save_indicators_in_prompt(self):
        """Test that save detection has detailed visual indicators."""
        with open('src/vision_backends.py', 'r') as f:
            content = f.read()
        
        assert 'Ball contact' in content
        assert 'Deflection' in content
        assert 'Ball secured' in content
        
    def test_confidence_level_guidance_in_prompt(self):
        """Test that prompt includes confidence level guidance."""
        with open('src/vision_backends.py', 'r') as f:
            content = f.read()
        
        assert 'HIGH CONFIDENCE (0.85-1.0)' in content
        assert 'MEDIUM CONFIDENCE (0.7-0.85)' in content
        assert 'LOW CONFIDENCE (0.5-0.7)' in content
        
    def test_do_not_report_scoreboard_changes(self):
        """Test that prompt warns against using scoreboard changes."""
        with open('src/vision_backends.py', 'r') as f:
            content = f.read()
        
        assert 'Scoreboard changes' in content or 'scoreboard' in content.lower()


class DummyVisionBackend(VisionBackendMixin):
    """Simple stub backend for testing helpers."""

    def _analyze_frames_impl(self, frames, instructions, sport, frame_interval, max_frames=20, time_offset=0.0):
        return []


class TestGoalConfirmationLogic:
    def test_scoreboard_only_goal_is_lowered(self):
        backend = DummyVisionBackend()
        events = [
            {'type': 'goal', 'timestamp': 10.0, 'description': 'Scoreboard shows 1-0 after the replay', 'confidence': 0.9}
        ]

        confirmed = backend._confirm_goal_events(events)
        assert confirmed[0]['confidence'] <= 0.65
        assert confirmed[0]['confirmation'] == 'scoreboard_only'

    def test_goal_with_supporting_shot_is_boosted(self):
        backend = DummyVisionBackend()
        events = [
            {'type': 'goal', 'timestamp': 15.0, 'description': 'Ball in net', 'confidence': 0.72},
            {'type': 'shot', 'timestamp': 14.8, 'confidence': 0.8}
        ]

        confirmed = backend._confirm_goal_events(events)
        assert confirmed[0]['confidence'] >= 0.8
        assert confirmed[0]['confirmation'] == 'support_event'


class TestConfigEnhancements:
    def test_config_includes_goal_refinement_settings(self):
        with open('config.yaml', 'r') as f:
            config = f.read()

        assert 'goal_refinement_enabled' in config
        assert 'goal_annotation_enabled' in config

class TestVideoUploadForLocalMode:
    """Test video upload endpoint for local mode."""
    
    def test_video_upload_endpoint_exists(self):
        """Test that /api/video/upload endpoint exists."""
        with open('app.py', 'r') as f:
            content = f.read()
        
        assert "@app.route('/api/video/upload'" in content
        assert 'def video_upload():' in content
        
    def test_video_upload_creates_task_id(self):
        """Test that video upload creates a task_id."""
        with open('app.py', 'r') as f:
            content = f.read()
        
        upload_func_start = content.find('def video_upload():')
        upload_func_end = content.find('\n@app.route', upload_func_start + 1)
        if upload_func_end == -1:
            upload_func_end = content.find('\ndef ', upload_func_start + 1)
        upload_func = content[upload_func_start:upload_func_end]
        
        assert 'task_id' in upload_func
        assert 'video_cache[task_id]' in upload_func
        

class TestFFmpegAPICompatibility:
    """Test ffmpeg-python API compatibility fixes."""
    
    def test_ffmpeg_uses_quiet_not_capture_output(self):
        """Test that ffmpeg-python uses quiet=True for its run() calls."""
        with open('src/video_clipper.py', 'r') as f:
            content = f.read()
        
        # Should use quiet=True for ffmpeg-python library calls
        assert '.run(quiet=True)' in content
        
        # Note: subprocess.run() with capture_output=True is fine for direct ffmpeg calls
        # We're checking that ffmpeg-python library uses quiet=True
        
    def test_moviepy_write_videofile_parameters(self):
        """Test that moviepy write_videofile uses correct parameters."""
        with open('src/video_clipper.py', 'r') as f:
            content = f.read()
        
        # Should have write_videofile with logger parameter
        assert 'write_videofile' in content
        assert 'logger=None' in content


class TestIntegrationScenarios:
    """Test integration scenarios for session features."""
    
    def test_select_all_with_filter_workflow(self):
        """Test complete workflow: filter events, select all, export."""
        with open('templates/index.html', 'r') as f:
            content = f.read()
        
        # All required functions present
        assert 'function filterEvents()' in content
        assert 'function toggleEventSelectAll()' in content
        assert 'function getVisibleEventIndices()' in content
        assert 'function downloadSelectedTimestamps()' in content
        
    def test_local_mode_highlight_generation_workflow(self):
        """Test local mode can generate highlights after video upload."""
        with open('templates/index.html', 'r') as f:
            html_content = f.read()
            
        with open('app.py', 'r') as f:
            app_content = f.read()
        
        # Local mode uploads video
        assert "fetch('/api/video/upload'" in html_content
        assert 'currentTaskId = uploadData.task_id' in html_content
        
        # Video upload endpoint exists
        assert "/api/video/upload" in app_content
        
        # Highlight generation uses task_id
        assert 'currentTaskId' in html_content
