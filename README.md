# Chat-Analyser


```markdown
# Chat Analysis Tool

## Overview

The Chat Analysis Tool is a Python script that analyzes text-based chat conversations. It provides insights into emoji usage, sentence occurrences, sentiment analysis, and word frequency.

## Features

- Extracts emojis from the chat text.
- Counts occurrences of specific sentences for both sender and receiver.
- Performs sentiment analysis on the entire chat.
- Analyzes word frequency and provides the top words.

## Prerequisites

- Python 3.x
- Required Python packages (install using `pip install package_name`):
  - emoji
  - nltk
  - Pillow
  - reportlab

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/chat-analysis-tool.git
   ```

2. Navigate to the project directory:

   ```bash
   cd chat-analysis-tool
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Export your chat as a text file (e.g., `exported_chat.txt`) and place it in the project directory.

5. Run the script:

   ```bash
   python chat_analysis.py
   ```

6. Follow the on-screen instructions to enter the sender and receiver sentences.

7. View the analysis results in the console.

## Emoji Images

Make sure to have a directory named `emoji_images` containing PNG images for the emojis used in the chat. The images should be named after their Unicode representation (e.g., `U0001F604.png`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Remember to customize the README to include specific details about your project, such as installation instructions, usage guidelines, and any additional features or configurations.