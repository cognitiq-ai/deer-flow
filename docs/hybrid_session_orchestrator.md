# Hybrid Session Orchestrator

The hybrid session orchestrator combines the best of both production-ready performance and development-friendly debugging capabilities. This approach allows you to use the same core logic with optional enhanced features when needed.

## Overview

The session orchestrator now supports two modes:

1. **Production Mode**: Optimized performance with basic logging
2. **Debug Mode**: Rich console output, progress tracking, and optional visualizations

## Key Benefits

### ✅ Production Ready
- **Zero overhead**: Debug features only load when explicitly enabled
- **Backward compatibility**: Existing code continues to work unchanged
- **Minimal dependencies**: Core functionality doesn't require UI libraries
- **Performance optimized**: No visualization overhead in production

### ✅ Developer Friendly
- **Rich console output**: Colors, progress bars, formatted panels
- **Real-time feedback**: See what's happening as it happens
- **Graph visualizations**: Optional interactive graphs with plotly
- **Enhanced error reporting**: Better debugging information
- **Modular design**: Easy to extend with new debug features

## Architecture

```
├── src/orchestrator/
│   ├── session.py              # Core orchestrator (unchanged logic)
│   ├── debug_utils.py          # Optional debug capabilities
│   └── models.py               # Shared data models
└── examples/
    └── hybrid_session_demo.py  # Usage examples
```

### Core Components

1. **`session_orchestrator()`** - Original production function (unchanged)
2. **`EnhancedSessionLogger`** - Session logger with optional debug callbacks
3. **`RichDebugCallbacks`** - Rich console implementation for debug features
4. **`session_orchestrator_with_debug()`** - Convenience wrapper for debug mode

## Usage Examples

### Production Mode (No Changes Required)

```python
from src.orchestrator.session import session_orchestrator

# Existing code continues to work exactly as before
result = await session_orchestrator(user_query_context_data)
```

### Debug Mode with Rich Output

```python
from src.orchestrator.session import session_orchestrator_with_debug

# Rich console output with progress tracking
result = await session_orchestrator_with_debug(
    user_query_context_data,
    enable_rich_output=True
)
```

### Interactive Mode with Visualizations

```python
from src.orchestrator.session import session_orchestrator_with_debug

# Full debug mode with interactive graph visualizations
result = await session_orchestrator_with_debug(
    user_query_context_data,
    enable_rich_output=True,
    show_interactive=True
)
```

### Synchronous Debug Mode

```python
from src.orchestrator.session import session_orchestrator_with_debug_sync

# No need for asyncio.run() - perfect for CLI tools
result = session_orchestrator_with_debug_sync(
    user_query_context_data,
    enable_rich_output=True
)
```

### Custom Console Configuration

```python
from rich.console import Console
from src.orchestrator.session import session_orchestrator_with_debug

# Custom console with specific settings
console = Console(width=120, force_terminal=True)
result = await session_orchestrator_with_debug(
    user_query_context_data,
    enable_rich_output=True,
    console=console
)
```

## Debug Features

### Rich Console Output

When `enable_rich_output=True`:

- **Session Start Panel**: Shows goal, topic, and prior knowledge level
- **Iteration Progress**: Real-time iteration counter and focus concepts
- **Inner Loop Tracking**: Individual concept processing status
- **Graph Statistics**: Tables showing nodes, relationships, and status counts
- **Progress Bars**: Visual progress indicators for batch processing
- **Colored Status**: ✅ ❌ 🔄 indicators for easy status recognition

### Graph Visualizations

When `show_interactive=True`:

- **Real-time Updates**: Graph visualizations update after each iteration
- **Interactive Plots**: Plotly-based graphs that open in browser
- **Statistics Tables**: Detailed breakdowns of graph structure
- **History Tracking**: See how the graph evolves over time

### Enhanced Error Reporting

- **Rich Exception Display**: Better formatted error messages
- **Context Information**: More detailed error context
- **Progress State**: Know exactly where errors occurred
- **Graceful Degradation**: Falls back to basic logging if rich unavailable

## Dependencies

### Core (Production)
- No additional dependencies beyond existing requirements

### Debug Mode (Optional)
- `rich`: For enhanced console output
- `plotly`: For interactive visualizations (if using `show_interactive=True`)

Install debug dependencies:
```bash
pip install rich plotly
```

## Migration Guide

### From CLI Debug Utility

If you were using the CLI debug utility, you can now get the same functionality with:

```python
# Old CLI approach
# python debug_cli.py --goal "Learn Python" --debug

# New hybrid approach
from src.orchestrator.session import session_orchestrator_with_debug_sync
from src.orchestrator.models import UserQueryContext

uqc = UserQueryContext(
    goal_string="Learn Python",
    raw_topic_string="Python",
    prior_knowledge_level="beginner"
)

result = session_orchestrator_with_debug_sync(
    uqc.model_dump(),
    enable_rich_output=True,
    show_interactive=True
)
```

### From Production Code

No changes required! Your existing production code continues to work:

```python
# This continues to work exactly as before
result = await session_orchestrator(user_query_context_data)
```

## API Integration

### Web API Example

```python
from fastapi import FastAPI
from src.orchestrator.session import session_orchestrator, session_orchestrator_with_debug

app = FastAPI()

@app.post("/kg/generate")
async def generate_knowledge_graph(request_data: dict, debug: bool = False):
    if debug:
        # Debug mode for development/testing
        return await session_orchestrator_with_debug(
            request_data,
            enable_rich_output=False,  # No rich output in API
            show_interactive=False
        )
    else:
        # Production mode for live traffic
        return await session_orchestrator(request_data)
```

### CLI Tool Example

```python
import click
from src.orchestrator.session import session_orchestrator_with_debug_sync

@click.command()
@click.option("--goal", prompt="Learning Goal")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--interactive", is_flag=True, help="Show interactive graphs")
def cli_tool(goal: str, debug: bool, interactive: bool):
    uqc_data = create_user_query_context(goal)
    
    result = session_orchestrator_with_debug_sync(
        uqc_data,
        enable_rich_output=debug,
        show_interactive=interactive
    )
    
    click.echo(f"Session completed: {result['additional_data']['overall_status']}")
```

## Configuration

The hybrid approach respects all existing configuration settings and adds optional debug-specific features through function parameters rather than configuration files.

### Debug Configuration

Debug features are controlled through function parameters:

- `enable_rich_output`: Enable/disable rich console output
- `show_interactive`: Enable/disable interactive visualizations
- `console`: Custom rich Console instance (optional)

### Production Configuration

All existing configuration continues to work through the `Configuration` class:

- `max_iteration_main`: Maximum iterations
- `max_parallel_inner_loops`: Parallel processing limits
- `enable_educational_content_generation`: Educational content phase
- All other existing settings

## Performance Considerations

### Production Mode
- **Zero overhead**: Debug code not loaded unless explicitly enabled
- **Same performance**: Identical to original implementation
- **Memory efficient**: No extra objects created for debug features

### Debug Mode
- **Minimal overhead**: Debug callbacks are lightweight
- **Optional dependencies**: Rich/plotly only loaded when needed
- **Graceful fallback**: Falls back to basic logging if dependencies missing

## Extension Points

The hybrid approach is designed to be easily extensible:

### Custom Debug Callbacks

```python
from src.orchestrator.debug_utils import DebugCallbacks, EnhancedSessionLogger

class CustomDebugCallbacks:
    def on_session_start(self, uqc):
        # Your custom logic here
        pass
    
    def on_awg_update(self, awg, iteration):
        # Custom graph analysis
        pass

# Use custom callbacks
custom_logger = EnhancedSessionLogger(debug_callbacks=CustomDebugCallbacks())
result = await session_orchestrator(uqc_data, session_logger=custom_logger)
```

### Custom Visualizations

```python
class MyVisualizationCallbacks(RichDebugCallbacks):
    def on_awg_update(self, awg, iteration):
        super().on_awg_update(awg, iteration)
        # Add your custom visualization logic
        self.save_graph_to_file(awg, f"iteration_{iteration}.png")
```

## Testing

The hybrid approach makes testing easier by providing better visibility into the process:

```python
import pytest
from src.orchestrator.session import session_orchestrator_with_debug

@pytest.mark.asyncio
async def test_knowledge_graph_generation():
    # Use debug mode for better test visibility
    result = await session_orchestrator_with_debug(
        test_uqc_data,
        enable_rich_output=True  # See what's happening in tests
    )
    
    assert result['additional_data']['overall_status'] == 'SUCCESS_PREREQUISITES_MET'
    assert len(result['session_metrics']['final_concept_count']) > 0
```

## Troubleshooting

### Common Issues

1. **Rich library not found**
   ```
   Warning: Rich library not available. Falling back to basic logging.
   ```
   Solution: `pip install rich` or use `enable_rich_output=False`

2. **Interactive graphs not showing**
   - Ensure `plotly` is installed: `pip install plotly`
   - Check that the knowledge graph has `show_interactive_graph` method

3. **Performance issues in debug mode**
   - Use `enable_rich_output=False` for production
   - Disable `show_interactive` for better performance

### Debug Information

Enable debug mode to get detailed information about:
- Session initialization and configuration
- Iteration progress and focus concepts
- Inner loop processing status
- AWG updates and graph statistics
- Educational content generation progress
- Error details and stack traces

## Best Practices

1. **Production**: Always use basic `session_orchestrator()` for production APIs
2. **Development**: Use `session_orchestrator_with_debug()` for development and testing
3. **CLI Tools**: Use `session_orchestrator_with_debug_sync()` for command-line tools
4. **Testing**: Enable rich output in tests for better visibility
5. **CI/CD**: Use production mode in automated pipelines
6. **Documentation**: Use interactive mode for demos and documentation
