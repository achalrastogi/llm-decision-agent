# Context-Aware LLM Model Comparison & Decision Agent

A multi-agent decision-support system that helps users choose between LLM models by explaining trade-offs based on their business constraints, operational requirements, and cost limits.

## 🎯 Project Overview

Unlike generic comparison tools that rely on static benchmarks and provide single "best model" answers, this system:

- **Extracts user constraints** through intelligent discovery
- **Eliminates incompatible options** with clear rationale
- **Compares remaining models** using weighted criteria
- **Explains why one option fits better** with detailed reasoning
- **Offers mitigation and future-proofing strategies**

## 🏗️ Architecture

### Multi-Agent System Design

The system employs a **sequential handoff pattern** with 5 core agents + 1 optional agent:

1. **🔍 Discovery Agent** - Extracts user intent, constraints, and priorities
2. **🔧 Constraint Filter Agent** - Eliminates incompatible models with rationale
3. **⚖️ Decision Engine Agent** - Applies weighted scoring and computes trade-offs
4. **💡 Recommendation Agent** - Converts analysis into human-readable advice
5. **🧪 Evaluation Agent** *(Optional)* - Runs A/B tests and behavioral checks

### Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python with explicit orchestration logic
- **Data Storage**: JSON/SQLite for session state
- **Model Registry**: YAML-based static registry
- **Dependencies**: Managed via `uv` package manager

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- `uv` package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd llm-decision-agent
   ```

2. **Create virtual environment**
   ```bash
   uv venv
   ```

3. **Activate virtual environment**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   uv add streamlit pydantic pyyaml pandas plotly
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

6. **Open in browser**
   - Navigate to `http://localhost:8501`

## 📋 Usage Guide

### Step-by-Step Workflow

1. **🔍 Discovery Phase**
   - Describe your LLM use case in natural language
   - Specify constraints (budget, latency, context window)
   - Set priority weights (reasoning, speed, cost, reliability)

2. **🔧 Filtering Phase**
   - Review eliminated models with detailed rationale
   - See constraint impact analysis
   - Understand which requirements filter out which models

3. **⚖️ Scoring Phase**
   - View weighted scores across multiple dimensions
   - Explore trade-off analysis between top models
   - Examine detailed score explanations

4. **💡 Recommendation Phase**
   - Get primary recommendation with confidence level
   - Review explicit trade-offs with alternatives
   - Access mitigation strategies and implementation guidance

### Example Use Cases

- **Analytical Tasks**: Research, data analysis, report generation
- **Generative Tasks**: Content creation, writing assistance, creative work
- **Agentic Tasks**: Automation, workflow orchestration, tool integration

## 📊 Model Registry

The system includes 4 example LLM models:

| Model | Provider | Context Window | Latency | Cost Tier |
|-------|----------|----------------|---------|-----------|
| GPT-4 | OpenAI | 128,000 | Interactive | Premium |
| Claude 3 Sonnet | Anthropic | 200,000 | Interactive | Standard |
| Gemini Pro | Google | 32,768 | Real-time | Budget |
| Llama 2 70B | Meta | 4,096 | Batch | Budget |

## 🔧 Configuration

### Model Registry

Models are defined in `models/model_registry.yaml`:

```yaml
models:
  model-id:
    name: "Model Name"
    provider: "Provider"
    context_window: 32768
    latency_category: "interactive"
    cost_tier: "standard"
    reasoning_strength: 8.5
    tool_reliability: 7.0
    capabilities: ["text_generation", "analysis"]
    deployment_options: ["api", "cloud"]
    cost_per_token:
      input: 0.000015
      output: 0.000075
    benchmark_scores:
      mmlu: 79.0
      hellaswag: 89.0
```

### Priority Weights

Default priority weights can be customized:

- **Reasoning**: Cognitive capabilities and accuracy
- **Latency**: Response time and throughput
- **Cost**: Token pricing and operational efficiency
- **Reliability**: API stability and tool integration

## 📁 Project Structure

```
llm-decision-agent/
├── .venv/                  # Virtual environment
├── agents/                 # Agent implementations
│   ├── discovery.py        # Constraint extraction
│   ├── constraint_filter.py # Model filtering
│   ├── decision_engine.py  # Scoring and trade-offs
│   └── recommendation.py   # Final recommendations
├── models/                 # Data models and registry
│   ├── model_registry.yaml # Model definitions
│   ├── registry.py         # Registry loader
│   └── schemas.py          # Pydantic schemas
├── ui/                     # Streamlit UI components
│   ├── discovery_page.py   # Discovery interface
│   ├── filtering_page.py   # Filtering results
│   ├── scoring_page.py     # Scoring visualization
│   └── recommendation_page.py # Final recommendations
├── main.py                 # Application entry point
├── pyproject.toml          # Dependencies
├── uv.lock                 # Dependency lock file
└── KNOWN_GAPS.md          # Identified limitations
```

## 🧪 Testing

### Manual Testing

1. **Discovery Agent**
   ```bash
   python -c "from agents.discovery import DiscoveryAgent; agent = DiscoveryAgent(); print(agent.extract_constraints('I need to analyze customer feedback quickly'))"
   ```

2. **Constraint Filter**
   ```bash
   python -c "from models.registry import ModelRegistry; from agents.constraint_filter import ConstraintFilterAgent; registry = ModelRegistry(); agent = ConstraintFilterAgent(registry); print(len(agent.filter_models(constraints)[0]))"
   ```

3. **End-to-End Workflow**
   - Run the Streamlit app and test complete user journey
   - Verify each phase produces expected results

## ⚠️ Known Limitations

### Critical Gaps

1. **No Actual LLM Usage**: Despite being an "AI agent" system, all logic is rule-based
2. **Static Model Data**: No real-time updates from provider APIs
3. **Limited Personalization**: Beyond basic priority weights
4. **Rule-Based Logic**: Cannot handle nuanced or ambiguous requirements

### See [KNOWN_GAPS.md](KNOWN_GAPS.md) for comprehensive limitations analysis.

## 🛣️ Roadmap

### Phase 1 (Current - MVP)
- ✅ Multi-agent architecture
- ✅ Constraint extraction and filtering
- ✅ Weighted scoring and recommendations
- ✅ Streamlit web interface

### Phase 2 (Next - AI Integration)
- 🔄 Integrate actual LLM calls for intelligent reasoning
- 🔄 Dynamic constraint understanding
- 🔄 Personalized recommendation generation
- 🔄 Real-time model data updates

### Phase 3 (Future - Production)
- ⏳ Multi-user support and authentication
- ⏳ API for external integration
- ⏳ Advanced analytics and monitoring
- ⏳ Enterprise deployment options

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and test thoroughly
4. Update KNOWN_GAPS.md if introducing new limitations
5. Submit a pull request

### Code Standards

- Follow PEP 8 for Python code style
- Use type hints for all function parameters and returns
- Add docstrings for all classes and methods
- Update tests when modifying functionality

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built following spec-driven development methodology
- Inspired by decision-support systems and multi-agent architectures
- Uses property-based testing principles for correctness validation

## 📞 Support

For questions, issues, or contributions:

1. Check [KNOWN_GAPS.md](KNOWN_GAPS.md) for known limitations
2. Search existing issues in the repository
3. Create a new issue with detailed description
4. For urgent matters, contact the development team

---

**Note**: This is currently an MVP implementation with rule-based logic. The next major iteration will integrate actual LLM capabilities for true AI agent behavior.