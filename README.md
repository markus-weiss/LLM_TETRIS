# Tetris with Self-Learning AI
## LLM_TETRIS
_PROJECT_LLM_TETRIS

This project is a Tetris game with a built-in self-learning AI, developed as an experimental test using ChatGPT. The AI utilizes reinforcement learning techniques to improve its gameplay over time.

## Features

- **Interactive Tetris Gameplay:** Classic Tetris mechanics with colorful Tetromino pieces.
- **Self-Learning AI:** An AI agent built using TensorFlow.js that learns how to play Tetris using reinforcement learning.
- **Dynamic Visualization:** Real-time game rendering on an HTML canvas.
- **Fully Responsive Design:** Adjusts to various screen sizes with a minimal and clean interface.

## Project Structure

```
📂 Project Root
├── index.html       # Main HTML file integrating the game canvas and scripts
├── style.css        # Styling for the Tetris interface
├── script.js        # Core game logic and AI implementation
```

## Technologies Used

- **HTML5**: For the game structure and canvas element.
- **CSS3**: For styling and layout.
- **JavaScript (ES6+)**: For game mechanics and AI logic.
- **TensorFlow.js**: For the AI model and training.

## How the AI Works

1. **State Representation:** The current game state, including the arena and the active Tetromino, is represented as a normalized input.
2. **Actions:** The AI chooses from four possible actions: move left, move right, rotate, or drop the Tetromino.
3. **Rewards:** The AI receives rewards for clearing lines and penalties for inefficient moves.
4. **Training:** The AI uses a deep Q-learning model with a memory buffer to improve over time.

## How to Run the Project

1. Clone the repository or download the files.
2. Ensure you have an internet connection to load TensorFlow.js from the CDN.
3. Open the `index.html` file in any modern browser.
4. The game will start automatically, and the AI will begin learning as it plays.

## Screenshots

_Example screenshots or gameplay GIFs can be added here._

## Future Improvements

- Implement a more sophisticated reward system for better AI optimization.
- Add a user interface to switch between AI and manual gameplay.
- Introduce difficulty levels and additional gameplay modes.

## License

This project is created as a test and is open for learning purposes. Feel free to modify and use the code as needed.

## Acknowledgments

This project was created with the assistance of ChatGPT by OpenAI and utilizes TensorFlow.js for AI development.

---

Enjoy exploring how AI learns to play Tetris!
