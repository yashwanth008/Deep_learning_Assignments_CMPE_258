# Multimodal AI with Google Gemini

Complete implementation of multimodal AI using Google Gemini API for text generation, image analysis, and advanced conversational AI.

**Video Walkthrough:** [https://drive.google.com/file/d/1ETklD4x_Rcyt0jtvQqorftZokog2HrxG/view?usp=sharing]
---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)


---

## Overview

This project demonstrates comprehensive use of Google Gemini's multimodal capabilities including:

- Advanced text-to-text conversation with reasoning
- Image analysis with multiple input methods
- Multimodal creative workflows
- Visual question answering
- DeepSeek R1 inspired chain-of-thought reasoning

---

## Features

### 1. Advanced Text Generation

- **Step-by-Step Reasoning** - DeepSeek R1 inspired problem solving
- **Creative Writing** - Story generation and content creation
- **Technical Explanations** - Complex concepts with analogies
- **Structured Outputs** - Analysis, reasoning, and conclusions

### 2. Image Analysis (3 Methods)

#### Method 1: Upload Your Own Images
```python
from google.colab import files
uploaded = files.upload()
```

#### Method 2: Generate Test Images
```python
from PIL import Image, ImageDraw
# Programmatically create test images
```

#### Method 3: Download from URLs
```python
import urllib.request
# Download and analyze from web sources
```

### 3. Multimodal Workflows

- Image → Analysis → Creative Content
- Combined vision and language processing
- Scene understanding and storytelling

### 4. Visual Question Answering

- Ask specific questions about images
- Spatial relationship understanding
- Contextual responses

---

## Technologies

### Core Framework
- **Google Gemini 3 Flash Preview** - Latest model (Feb 2026)
- **Google Gemini 2.5 Flash** - Stable production model
- **Python 3.10+**
- **Google Colab** environment

### Libraries
```
google-generativeai >= 0.3.0
pillow >= 10.0.0
matplotlib >= 3.7.0
```

---

## Installation

### Prerequisites

1. **Google Colab Account** (free)
   - Visit: https://colab.research.google.com/

2. **Gemini API Key**
   - Get yours: https://makersuite.google.com/app/apikey

### Setup Steps

**Step 1: Upload Notebook**
```bash
# Upload multimodal_gemini_demo.ipynb to Google Colab
```

**Step 2: Configure API Key**
1. Click the key icon  in Colab's left sidebar
2. Add secret: `GEMINI_API_KEY`
3. Paste your API key value

**Step 3: Install Dependencies**
```python
# Run this cell in the notebook
!pip install -q google-generativeai pillow matplotlib
```

**Step 4: Run the Notebook**
- Runtime → Run all
- Or run cells individually (Shift + Enter)

---

## Usage

### Quick Start

```python
# 1. Import and configure
import google.generativeai as genai
genai.configure(api_key='YOUR_API_KEY')

# 2. Initialize model
model = genai.GenerativeModel('gemini-2.5-flash')

# 3. Text generation
response = model.generate_content("Explain quantum computing")
print(response.text)

# 4. Image analysis
from PIL import Image
img = Image.open('photo.jpg')
response = model.generate_content(["Describe this image", img])
print(response.text)
```

### Part 1: Text Conversations

Run cells to see:
- Complex reasoning with step-by-step solutions
- Creative story writing
- Technical explanations

**Example:**
```python
prompt = """
Think step-by-step:
If I have 3 apples and buy 2 packs of 5 apples each,
then give half to my friend, how many do I have?
"""
response = model.generate_content(prompt)
```

### Part 2: Image Upload

```python
from google.colab import files

# Upload image
uploaded = files.upload()
image_name = list(uploaded.keys())[0]
img = Image.open(image_name)

# Analyze
response = model.generate_content([
    "Analyze this image in detail",
    img
])
```

### Part 3: Generated Images

The notebook creates test images automatically:
```python
from PIL import Image, ImageDraw

img = Image.new('RGB', (800, 600))
draw = ImageDraw.Draw(img)
# Create artistic scene...
```

### Part 4: Multimodal Workflow

```python
# Step 1: Generate scene
scene_img = create_landscape()

# Step 2: Analyze
analysis = model.generate_content(["Describe this scene", scene_img])

# Step 3: Create story from analysis
story_prompt = f"Write a story based on: {analysis.text}"
story = model.generate_content(story_prompt)
```

---

## Project Structure

```
multimodal-gemini-project/
├── multimodal_gemini_demo.ipynb    # Main Colab notebook
├── README.md                        # This file
├── requirements.txt                 # Python dependencies

```

---

## Examples

### Example 1: Reasoning Task

**Input:**
```
Problem: A company has 3 departments with 15, 22, and 13 people.
Buses hold 25 people. How many buses needed?
```

**Output:**
```
Initial Analysis: Total people = 15 + 22 + 13 = 50

Step-by-step:
1. Calculate total: 50 people
2. Divide by capacity: 50 ÷ 25 = 2
3. Check remainder: 0 people remaining

Final Answer: 2 buses exactly
```

### Example 2: Image Analysis

**Input:** Photo of sunset

**Output:**
```
1. Main subject: Sunset over ocean with orange/pink sky
2. Colors: Warm oranges, deep purples, calm blues
3. Mood: Peaceful, contemplative, serene
4. Details: Sun partially below horizon, scattered clouds
5. Uses: Meditation apps, travel blogs, wall art
```

### Example 3: Visual Q&A

**Question:** "How many red shapes are in this image?"

**Answer:** "There are 2 red shapes: a red circle in the upper left and a red triangle in the lower right."

---

## Troubleshooting

### API Key Issues

**Problem:** `Invalid API key`

**Solution:**
```python
# Check key is in secrets
from google.colab import userdata
key = userdata.get('GEMINI_API_KEY')
print(f"Key starts with: {key[:10]}...")
```

### Image Upload Fails

**Problem:** Upload dialog doesn't appear

**Solution:**
1. Restart runtime (Runtime → Restart runtime)
2. Try different browser (Chrome recommended)
3. Use generated images instead

### Model Not Found

**Problem:** `404 models/gemini-pro not found`

**Solution:**
```python
# List available models
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(model.name)

# Use current model
model = genai.GenerativeModel('gemini-2.5-flash')
```

### URL Download Errors

**Problem:** `HTTP Error 404`

**Solution:**
- Use the automatic fallback in the notebook
- Upload images directly instead
- Use generated test images

### Memory Issues

**Problem:** Runtime crashes

**Solution:**
```python
# Process images one at a time
# Clear variables between runs
del large_image
import gc
gc.collect()
```

---

## Performance

### Speed Benchmarks

| Task | Time | Notes |
|------|------|-------|
| Text Generation | 2-5s | Simple queries |
| Image Analysis | 3-7s | Standard photos |
| Complex Reasoning | 5-10s | Multi-step problems |
| Full Notebook | 2-3min | All cells |

### Accuracy

- **Text Understanding:** Excellent
- **Image Recognition:** Very High
- **Reasoning:** Strong logical capabilities
- **Creative Output:** High quality and coherent

### Resource Usage

- **RAM:** ~2GB typical
- **Storage:** Minimal (images in memory)
- **Network:** ~1-5MB per request

---

## Models Comparison

### Gemini 3 Flash Preview

**Strengths:**
- Latest features and capabilities
- Enhanced reasoning
- Faster response times
- Best for experimentation

**Limitations:**
- Preview/experimental status
- May change behavior
- Not guaranteed stability

### Gemini 2.5 Flash

**Strengths:**
- Production-ready
- Stable and reliable
- Long-term support
- Recommended for applications

**Limitations:**
- Slightly older capabilities
- May not have newest features

---



