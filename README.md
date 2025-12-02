# 🚀 Multimodal Fashion Assistant

A cutting-edge fashion assistant leveraging multimodal AI for personalized style recommendations and interactive fashion experiences.

Your personal AI stylist, combining visual search, attribute filtering, and conversational AI to revolutionize your wardrobe.

##

![License](https://img.shields.io/github/license/Abrar-Islam-Oitijjho/Multimodal-Fashion-Assistant)
![GitHub stars](https://img.shields.io/github/stars/Abrar-Islam-Oitijjho/Multimodal-Fashion-Assistant?style=social)
![GitHub forks](https://img.shields.io/github/forks/Abrar-Islam-Oitijjho/Multimodal-Fashion-Assistant?style=social)
![GitHub issues](https://img.shields.io/github/issues/Abrar-Islam-Oitijjho/Multimodal-Fashion-Assistant)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Abrar-Islam-Oitijjho/Multimodal-Fashion-Assistant)
![GitHub last commit](https://img.shields.io/github/last-commit/Abrar-Islam-Oitijjho/Multimodal-Fashion-Assistant)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-%23FF5800.svg?style=for-the-badge&logo=Transformers&logoColor=white)

## 📋 Table of Contents

- [About](#about)
- [Features](#features)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Testing](#testing)
- [Deployment](#deployment)
- [FAQ](#faq)
- [License](#license)
- [Support](#support)
- [Acknowledgments](#acknowledgments)

## About

The Multimodal Fashion Assistant is a retrieval and reasoning system built on two models: SigLIP2-base for image and text embeddings, and Qwen-VL (2B) for multimodal reasoning. The system processes images and text to generate meaningful representations, runs similarity search, and handles attribute-based queries. It can operate in multimodal mode or as a single-mode (image-only or text-only) retrieval tool.

The pipeline uses SigLIP2 to extract high-quality embeddings from user input, while Qwen-VL interprets queries, extracts product attributes, and resolves complex fashion-related questions. Users can upload an image, describe a product, or refine their request through follow-up prompts. The system then returns the closest matching items based on visual and semantic similarity.

This project focuses on practical, fast, and accurate fashion retrieval without requiring a dedicated front end. It acts as a flexible backend module that supports image search, text search, and multimodal reasoning for fashion data.

##  Features

- 🎯 **Similarity Search**: Find visually similar clothing items using image embeddings.
- ⚡ **Attribute Filtering**: Filter fashion items based on specific attributes like color, style, material and more.
- 🤖 **Interactive Conversations**: Engage in multi-turn conversations to refine your fashion search.
- 🎨 **Personalized Recommendations**: Receive tailored style suggestions based on your preferences.
- 🛠️ **Extensible Architecture**: Easily integrate new models and features.

##  Demo

🔗 **Live Demo**: [https://your-demo-url.com](https://your-demo-url.com)

### Screenshots
![Fashion Search Interface](screenshots/fashion_search.png)
*Main interface showing image upload and text search options*

![Attribute Filtering](screenshots/attribute_filter.png)
*Attribute filtering options for refining search results*

![Conversation Interface](screenshots/conversation_ui.png)
*Interactive conversation UI for personalized recommendations*

##  Quick Start

Clone and run the Jupyter Notebook:

```bash
git clone https://github.com/Abrar-Islam-Oitijjho/Multimodal-Fashion-Assistant.git
cd Multimodal-Fashion-Assistant
jupyter notebook Multimodal_Fashion_Assistant.ipynb
```

Open the notebook in your browser and follow the instructions.

##  Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- PyTorch
- Transformers
- Other dependencies listed in `requirements.txt`

### Steps

1.  Clone the repository:

```bash
git clone https://github.com/Abrar-Islam-Oitijjho/Multimodal-Fashion-Assistant.git
cd Multimodal-Fashion-Assistant
```

2.  Create a virtual environment using Anaconda Prompt (recommended):

```bash
conda create -n venv python=3.11 anaconda
conda activate venv
conda install ipykernel
ipython kernel install --user --name=kernelname
```

3.  Install dependencies:

```bash
pip install -r requirements.txt
```

##  Usage

1.  Open the Jupyter Notebook:

```bash
conda activate venv
jupyter notebook
```

2. Then find the notebook "multimodal_reasoning.ipynb" and open it.

3.  Follow the instructions within the notebook to load models, process data, and interact with the fashion assistant.

### Example: Image Similarity Search

```python
# Example code (Illustrative - refer to notebook for actual implementation)
user_query = "Do you have any dress like this?"

image_directory = r"../Data/Example"
image_path = os.path.join(image_directory, "internet_example13.jpg")
image = Image.open(image_path).convert("RGB")

conversation = []
output_text = chtbot.generate_description(conversation=conversation, user_query=user_query, user_image=image)
print(output_text[0])
```

### Example Output

The user wants a red and black plaid dress with a Peter Pan collar. The dress has a fit-and-flare silhouette and long sleeves. It is made of a material that appears to be a blend of cotton and polyester. The dress features a button-down front and a waistband with a belt. The notable features of the dress include the Peter Pan collar, the fit-and-flare silhouette.

<p>
  <img src="assets/output_0_0.jpg" width="180">
  <img src="assets/output_0_1.jpg" width="180">
  <img src="assets/output_0_2.jpg" width="180">
  <img src="assets/output_0_3.jpg" width="180">
  <img src="assets/output_0_4.jpg" width="180">
  <img src="assets/output_0_5.jpg" width="180">
</p>


### Example: Conversational Interaction

```python
# Example code (Illustrative - refer to notebook for actual implementation)
response = assistant.chat("I'm looking for a red dress for a party.")
print(response)
```

##  Configuration

### Environment Variables

Create a `.env` file (optional) in the root directory to configure API keys and other settings:

```env
# API Keys
IMAGE_API_KEY=your_image_api_key
TEXT_API_KEY=your_text_api_key
```

### Configuration File (Example)

```json
{
  "model_settings": {
    "siglip2_model": "path/to/siglip2/model",
    "qwen_vl_model": "path/to/qwen_vl/model"
  },
  "search_parameters": {
    "top_k": 10,
    "attribute_weights": {
      "color": 0.5,
      "style": 0.3,
      "material": 0.2
    }
  }
}
```

##  Project Structure

```
Multimodal-Fashion-Assistant/
├── 📁 data/                   # Fashion dataset and image files
├── 📁 models/                 # Pre-trained model weights
├── 📁 notebooks/              # Jupyter notebooks for experimentation
├── 📁 src/                    # Source code for the fashion assistant
│   ├── 📄 fashion_assistant.py  # Core class for the assistant
│   ├── 📄 utils.py              # Utility functions
│   └── 📄 api.py                # API endpoints (if applicable)
├── 📄 requirements.txt        # Project dependencies
├── 📄 README.md               # Project documentation
└── 📄 LICENSE                 # License file
```

##  Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) (placeholder) for details.

### Quick Contribution Steps

1.  🍴 Fork the repository
2.  🌟 Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  ✅ Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  📤 Push to the branch (`git push origin feature/AmazingFeature`)
5.  🔃 Open a Pull Request

## Testing

Testing instructions and commands will be added here.

## Deployment

Deployment instructions will be added here.

## FAQ

Frequently asked questions will be added here.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### License Summary

-   ✅ Commercial use
-   ✅ Modification
-   ✅ Distribution
-   ✅ Private use
-   ❌ Liability
-   ❌ Warranty

##  Support

-   📧 **Email**: your.email@example.com
-   💬 **Discord**: [Join our community](https://discord.gg/your-invite)
-   🐛 **Issues**: [GitHub Issues](https://github.com/Abrar-Islam-Oitijjho/Multimodal-Fashion-Assistant/issues)
-   📖 **Documentation**: [Full Documentation](https://docs.your-site.com)
-   💰 **Sponsor**: [Support the project](https://github.com/sponsors/Abrar-Islam-Oitijjho)

##  Acknowledgments

-   🎨 **Design inspiration**: [Dribbble](https://dribbble.com/)
-   📚 **Libraries used**:
    -   [PyTorch](https://pytorch.org/) - Deep learning framework
    -   [Transformers](https://huggingface.co/transformers/) - NLP library
-   👥 **Contributors**: Thanks to all [contributors](https://github.com/Abrar-Islam-Oitijjho/Multimodal-Fashion-Assistant/contributors)
-   🌟 **Special thanks**: To the open-source community for their invaluable contributions.
```
