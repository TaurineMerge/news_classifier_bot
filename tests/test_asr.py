def test_asr_module_imports_without_loading_model():
    from newsbot.asr import WhisperASR

    assert callable(WhisperASR)
