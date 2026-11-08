import sys
from unittest.mock import MagicMock
import pytest

@pytest.fixture(scope="module", autouse=True)
def mock_dependencies():
    # Mock dependencies
    mock_gym = MagicMock()
    class MockEnv: pass
    mock_gym.Env = MockEnv

    # Save original modules to restore later
    original_modules = {
        'gymnasium': sys.modules.get('gymnasium'),
        'gymnasium.spaces': sys.modules.get('gymnasium.spaces'),
        'numpy': sys.modules.get('numpy'),
        'astropy': sys.modules.get('astropy'),
        'astropy.time': sys.modules.get('astropy.time'),
        'astropy.coordinates': sys.modules.get('astropy.coordinates'),
        'astropy.units': sys.modules.get('astropy.units'),
        'joblib': sys.modules.get('joblib'),
        'pandas': sys.modules.get('pandas'),
    }

    sys.modules['gymnasium'] = mock_gym
    sys.modules['gymnasium.spaces'] = MagicMock()
    sys.modules['astropy'] = MagicMock()
    sys.modules['astropy.time'] = MagicMock()
    sys.modules['astropy.coordinates'] = MagicMock()
    sys.modules['astropy.units'] = MagicMock()
    sys.modules['joblib'] = MagicMock()
    sys.modules['pandas'] = MagicMock()
    sys.modules['numpy'] = MagicMock()

    yield

    # Restore original modules
    for name, module in original_modules.items():
        if module is None:
            if name in sys.modules:
                del sys.modules[name]
        else:
            sys.modules[name] = module

def test_normalize_scalar():
    # Import inside test to ensure mocks are active
    from spacecraft_env import SpacecraftEnv

    # Test case 1: Standard values
    # (5 - 0) / (10 - 0) = 0.5
    assert SpacecraftEnv._normalize(None, 5, 0, 10) == 0.5

    # Test case 2: Min boundary
    assert SpacecraftEnv._normalize(None, 0, 0, 10) == 0.0

    # Test case 3: Max boundary
    assert SpacecraftEnv._normalize(None, 10, 0, 10) == 1.0

    # Test case 4: Negative range
    # (-5 - (-10)) / (0 - (-10)) = 5 / 10 = 0.5
    assert SpacecraftEnv._normalize(None, -5, -10, 0) == 0.5

def test_normalize_negative_values():
    from spacecraft_env import SpacecraftEnv
    # (-2 - (-5)) / (5 - (-5)) = 3 / 10 = 0.3
    assert SpacecraftEnv._normalize(None, -2, -5, 5) == 0.3

def test_normalize_large_values():
    from spacecraft_env import SpacecraftEnv
    # (150 - 100) / (200 - 100) = 50 / 100 = 0.5
    assert SpacecraftEnv._normalize(None, 150, 100, 200) == 0.5
