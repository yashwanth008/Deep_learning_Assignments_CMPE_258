# Pixiee - AI-Powered Full-Stack Todo Application
**Video Walkthrough** [https://drive.google.com/file/d/147gmSq9P1CaDyxiUwDmJUufBcQCsOZly/view?usp=sharing]

![project-image](<project.png>)
##  Overview
A modern, AI-enhanced full-stack todo application built using agentic AI tools (Antigravity/Claude Code). Features intelligent task suggestions, beautiful UI, and complete task management capabilities.

## Features

### Core Features
-  **Create, Read, Update, Delete (CRUD)** operations for tasks
-  **Category Management** - Organize tasks by Work, Personal, Shopping, Health
-  **Task Completion Tracking** with visual feedback
-  **Beautiful Gradient UI** with smooth animations
-  **LocalStorage Persistence** - Tasks saved automatically
-  **Real-time Statistics** - Track total, active, and completed tasks

### AI-Powered Features
-  **AI Productivity Suggestions** - Smart tips that rotate every 30 seconds
-  **Intelligent Task Categorization** - Pre-defined categories for better organization
-  **Progress Tracking** - Visual indicators of task completion

### UI/UX Features
-  **Smooth Animations** - Fade-in, slide-in, and hover effects
-  **Color-Coded Categories** - Visual distinction for different task types
-  **Responsive Design** - Works on all screen sizes
-  **Gradient Theme** - Modern purple gradient aesthetic
-  **Advanced Filtering** - Filter by status or category

##  Technologies Used

### Frontend
- **React 18** - UI framework
- **Vanilla CSS** - Custom styling with gradients and animations
- **LocalStorage API** - Data persistence

### AI Tools Used
- **Google Antigravity** / **Claude Code** - Agentic AI assistance
- **wshobson Skills Package** - Best practices and patterns

### Deployment
- **Firebase Hosting** - Free, fast hosting solution
- **GitHub Pages** (alternative) - Version control integration

##  Installation & Setup

### Prerequisites
```bash
# Node.js 16+ and npm
node --version
npm --version

# Firebase CLI (for deployment)
npm install -g firebase-tools
```

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/[your-username]/ai-coding-assignment-2026.git
   cd ai-coding-assignment-2026/02-fullstack-web-app
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Run Locally**
   ```bash
   npm start
   ```
   Open http://localhost:3000 in your browser

4. **Build for Production**
   ```bash
   npm run build
   ```

##  Firebase Deployment

### Step-by-Step Deployment

1. **Login to Firebase**
   ```bash
   firebase login
   ```

2. **Initialize Firebase Project**
   ```bash
   firebase init
   ```
   - Select "Hosting"
   - Choose or create a Firebase project
   - Set public directory to "public"
   - Configure as single-page app: Yes
   - Don't overwrite index.html

3. **Deploy**
   ```bash
   firebase deploy
   ```

4. **Access Your App**
   ```
   https://[your-project-id].web.app
   ```

### Alternative: GitHub Pages

```bash
# Add to package.json
"homepage": "https://[username].github.io/pixiee-todo"

# Deploy
npm run build
git add .
git commit -m "Deploy to GitHub Pages"
git push origin main
```

##  Project Structure

```
02-fullstack-web-app/
├── public/
│   └── index.html          # Main app file (React + CSS)
├── package.json            # Dependencies and scripts
├── firebase.json           # Firebase configuration
├── .firebaserc            # Firebase project settings
└── README.md              # This file
```

##  Code Overview

### React Components

The app uses a single-component architecture with React Hooks:

```javascript
// State Management
const [todos, setTodos] = useState([]);          // Todo items
const [inputValue, setInputValue] = useState(''); // Input field
const [category, setCategory] = useState('personal'); // Selected category
const [filter, setFilter] = useState('all');     // Active filter

// Core Functions
addTodo()       // Create new task
toggleTodo()    // Mark complete/incomplete
deleteTodo()    // Remove task
getFilteredTodos() // Apply filters
```

### Data Structure

```javascript
{
  id: 1644567890123,           // Unique timestamp ID
  text: "Complete assignment", // Task description
  completed: false,            // Completion status
  category: "work",            // Category (work/personal/shopping/health)
  createdAt: "2026-02-14T..."  // ISO timestamp
}
```

### AI Suggestion System

```javascript
// Rotates productivity tips every 30 seconds
const suggestions = [
  " Break large tasks into smaller steps!",
  " Focus on 3 important tasks today",
  " Set specific time blocks",
  // ... more suggestions
];
```

## Design System

### Color Palette
- **Primary Gradient:** `#667eea` → `#764ba2`
- **Work:** `#f44336` (Red)
- **Personal:** `#2196f3` (Blue)
- **Shopping:** `#4caf50` (Green)
- **Health:** `#ff9800` (Orange)

### Animations
- **fadeInDown** - Header entrance
- **fadeInUp** - Card entrance
- **slideIn** - Todo item entrance
- **pulse** - AI suggestion highlight

##  Testing the App

### Manual Testing Checklist
- [ ] Add new task
- [ ] Mark task as complete
- [ ] Delete task
- [ ] Filter by category
- [ ] Filter by status (active/completed)
- [ ] Refresh page (persistence test)
- [ ] Test on mobile device
- [ ] Test AI suggestions rotation

### Browser Compatibility
-  Chrome/Edge 90+
-  Firefox 88+
-  Safari 14+
-  Mobile browsers

##  Features Breakdown

| Feature | Technology | Complexity |
|---------|-----------|------------|
| Task CRUD | React Hooks | Medium |
| Data Persistence | LocalStorage | Easy |
| Filtering System | Array methods | Medium |
| AI Suggestions | JavaScript Timer | Easy |
| Animations | CSS Keyframes | Medium |
| Responsive Design | CSS Media Queries | Medium |

##  Future Enhancements

- [ ] User authentication with Firebase Auth
- [ ] Cloud sync with Firestore
- [ ] Drag-and-drop task reordering
- [ ] Due dates and reminders
- [ ] Task priority levels
- [ ] Dark mode toggle
- [ ] Export to CSV/PDF
- [ ] Real AI integration (GPT-4)
- [ ] Collaboration features
- [ ] Mobile app (React Native)

