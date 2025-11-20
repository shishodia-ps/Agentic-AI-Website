// Simplified version to test topic loading
class AIKnowledgeHub {
    constructor() {
        this.topics = this.getDefaultTopics();
        this.init();
    }

    init() {
        console.log('AIKnowledgeHub initialized with', this.topics.length, 'topics');
        this.renderTopicGrid();
        this.renderFeaturedTopics();
        this.setupEventListeners();
    }

    getDefaultTopics() {
        return [
            {
                id: 1,
                title: "Introduction to Generative AI",
                category: "fundamentals",
                difficulty: "beginner",
                description: "Learn the fundamentals of generative AI, how Large Language Models work, and their core capabilities in content generation.",
                concepts: ["Generative AI", "LLMs", "Content Generation", "AI Fundamentals"],
                featured: true,
                progress: 0,
                content: "Introduction to Generative AI content..."
            },
            {
                id: 2,
                title: "Exploring and Comparing LLMs",
                category: "fundamentals",
                difficulty: "beginner",
                description: "Understand different types of Large Language Models, compare their capabilities, and learn how to select the right model for your use case.",
                concepts: ["Model Comparison", "GPT", "Claude", "Llama", "Model Selection"],
                featured: true,
                progress: 0,
                content: "Exploring and Comparing LLMs content..."
            },
            {
                id: 3,
                title: "Using Generative AI Responsibly",
                category: "fundamentals",
                difficulty: "beginner",
                description: "Learn the principles of Responsible AI, ethical considerations, and best practices for building trustworthy AI applications.",
                concepts: ["Responsible AI", "Ethics", "Bias", "Fairness", "Transparency"],
                featured: false,
                progress: 0,
                content: "Using Generative AI Responsibly content..."
            },
            {
                id: 4,
                title: "Prompt Engineering Fundamentals",
                category: "implementation",
                difficulty: "beginner",
                description: "Master the art and science of crafting effective prompts to guide AI models toward desired outputs and behaviors.",
                concepts: ["Prompt Engineering", "Zero-shot", "Few-shot", "Chain-of-Thought", "System Prompts"],
                featured: true,
                progress: 0,
                content: "Prompt Engineering Fundamentals content..."
            },
            {
                id: 5,
                title: "Retrieval-Augmented Generation (RAG)",
                category: "implementation",
                difficulty: "intermediate",
                description: "Learn how to enhance LLM responses with external knowledge using RAG techniques and vector databases.",
                concepts: ["RAG", "Vector Databases", "Embeddings", "Semantic Search", "Knowledge Retrieval"],
                featured: true,
                progress: 0,
                content: "Retrieval-Augmented Generation content..."
            },
            {
                id: 6,
                title: "Building AI Agents",
                category: "architecture",
                difficulty: "intermediate",
                description: "Create autonomous AI agents that can perceive, reason, act, and learn to accomplish complex tasks.",
                concepts: ["AI Agents", "Autonomous Systems", "Tool Use", "Planning", "Multi-Agent Systems"],
                featured: true,
                progress: 0,
                content: "Building AI Agents content..."
            },
            {
                id: 7,
                title: "LangChain Framework",
                category: "frameworks",
                difficulty: "intermediate",
                description: "Master LangChain for building applications with language models using chains, agents, and memory systems.",
                concepts: ["LangChain", "Chains", "Agents", "Memory", "Tools", "Prompts"],
                featured: true,
                progress: 0,
                content: "LangChain Framework content..."
            },
            {
                id: 8,
                title: "Vector Databases for AI Applications",
                category: "architecture",
                difficulty: "intermediate",
                description: "Learn about vector databases, embeddings, and semantic search for building scalable AI applications.",
                concepts: ["Vector Databases", "Embeddings", "Semantic Search", "Similarity Search", "Pinecone", "Chroma"],
                featured: false,
                progress: 0,
                content: "Vector Databases content..."
            },
            {
                id: 9,
                title: "Fine-Tuning Large Language Models",
                category: "advanced",
                difficulty: "advanced",
                description: "Learn advanced techniques for fine-tuning LLMs on custom datasets to improve performance on specific tasks.",
                concepts: ["Fine-tuning", "Transfer Learning", "Custom Datasets", "PEFT", "LoRA", "QLoRA"],
                featured: false,
                progress: 0,
                content: "Fine-Tuning LLMs content..."
            },
            {
                id: 10,
                title: "AutoGen Multi-Agent Framework",
                category: "frameworks",
                difficulty: "advanced",
                description: "Build sophisticated multi-agent systems with AutoGen's conversational approach to AI agent orchestration.",
                concepts: ["AutoGen", "Multi-Agent Systems", "Conversational AI", "Agent Orchestration", "Microsoft Research"],
                featured: true,
                progress: 0,
                content: "AutoGen Framework content..."
            }
        ];
    }

    renderTopicGrid() {
        const topicGrid = document.getElementById('topicGrid');
        if (!topicGrid) {
            console.log('Topic grid not found');
            return;
        }

        console.log('Rendering topic grid with', this.topics.length, 'topics');
        
        const categories = this.groupTopicsByCategory();
        topicGrid.innerHTML = '';

        Object.entries(categories).forEach(([category, topics]) => {
            const categoryCard = this.createCategoryCard(category, topics);
            topicGrid.appendChild(categoryCard);
        });
    }

    createCategoryCard(category, topics) {
        const card = document.createElement('div');
        card.className = 'topic-card stagger-item';
        
        const categoryInfo = this.getCategoryInfo(category);
        const difficultyCounts = this.getDifficultyCounts(topics);
        
        card.innerHTML = `
            <div class="topic-icon">
                ${categoryInfo.icon}
            </div>
            <div class="difficulty-badge difficulty-${topics[0].difficulty}">
                ${topics.length} topics
            </div>
            <h3 class="text-xl font-bold text-charcoal-warm mb-3">${categoryInfo.name}</h3>
            <p class="text-gray-warm mb-4">${categoryInfo.description}</p>
            <div class="flex flex-wrap gap-2 mb-4">
                ${Object.entries(difficultyCounts).map(([diff, count]) => 
                    `<span class="difficulty-badge difficulty-${diff}">${count} ${diff}</span>`
                ).join('')}
            </div>
            <div class="flex items-center justify-between">
                <span class="text-sm text-gray-warm">
                    ${Math.round(topics.reduce((sum, t) => sum + t.progress, 0) / topics.length)}% complete
                </span>
                <button class="text-copper-accent hover:text-sage-deep font-semibold" onclick="window.location.href='topics.html?category=${category}'">
                    Explore →
                </button>
            </div>
        `;

        return card;
    }

    renderFeaturedTopics() {
        const carousel = document.getElementById('featuredCarousel');
        if (!carousel) return;

        const featuredTopics = this.topics.filter(topic => topic.featured);
        carousel.innerHTML = '';

        featuredTopics.forEach(topic => {
            const card = this.createFeaturedCard(topic);
            carousel.appendChild(card);
        });
    }

    createFeaturedCard(topic) {
        const card = document.createElement('div');
        card.className = 'flex-shrink-0 w-80 bg-white rounded-2xl p-6 border border-gray-100 hover:border-copper-accent/30 transition-all duration-300 cursor-pointer';
        
        card.innerHTML = `
            <div class="difficulty-badge difficulty-${topic.difficulty} mb-3">
                ${topic.difficulty}
            </div>
            <h3 class="text-lg font-bold text-charcoal-warm mb-2">${topic.title}</h3>
            <p class="text-gray-warm text-sm mb-4 line-clamp-3">${topic.description}</p>
            <div class="flex flex-wrap gap-1 mb-4">
                ${topic.concepts.slice(0, 3).map(concept => 
                    `<span class="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">${concept}</span>`
                ).join('')}
            </div>
            <div class="flex items-center justify-between">
                <span class="text-xs text-gray-500">${topic.category}</span>
                <button class="text-copper-accent hover:text-sage-deep text-sm font-semibold" onclick="window.aiHub.viewTopic(${topic.id})">
                    Learn More →
                </button>
            </div>
        `;

        card.addEventListener('click', () => this.viewTopic(topic.id));
        return card;
    }

    setupEventListeners() {
        // Search functionality
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => this.handleSearch(e.target.value));
        }

        // Modal functionality
        const addTopicBtn = document.getElementById('addTopicBtn');
        if (addTopicBtn) {
            addTopicBtn.addEventListener('click', () => this.showModal());
        }
    }

    handleSearch(query) {
        console.log('Searching for:', query);
        // Implement search functionality
    }

    showModal() {
        console.log('Show modal called');
        // Implement modal functionality
    }

    viewTopic(topicId) {
        console.log('Viewing topic:', topicId);
        // Store the topic ID and navigate to topics page
        localStorage.setItem('selectedTopicId', topicId);
        window.location.href = 'topics.html';
    }

    // Utility functions
    groupTopicsByCategory() {
        return this.topics.reduce((groups, topic) => {
            if (!groups[topic.category]) {
                groups[topic.category] = [];
            }
            groups[topic.category].push(topic);
            return groups;
        }, {});
    }

    getCategoryInfo(category) {
        const categoryMap = {
            'fundamentals': {
                name: 'AI Fundamentals',
                description: 'Core concepts and basic principles of AI agents',
                icon: '🧠'
            },
            'architecture': {
                name: 'Agent Architecture',
                description: 'Technical design patterns and system architectures',
                icon: '🏗️'
            },
            'frameworks': {
                name: 'Frameworks & Tools',
                description: 'Development frameworks and implementation tools',
                icon: '🛠️'
            },
            'implementation': {
                name: 'Implementation',
                description: 'Practical development guides and best practices',
                icon: '⚡'
            },
            'advanced': {
                name: 'Advanced Concepts',
                description: 'Complex theoretical topics and research areas',
                icon: '🚀'
            },
            'applications': {
                name: 'Industry Applications',
                description: 'Real-world use cases and industry implementations',
                icon: '🏭'
            }
        };
        
        return categoryMap[category] || { name: category, description: 'AI topics', icon: '📚' };
    }

    getDifficultyCounts(topics) {
        return topics.reduce((counts, topic) => {
            counts[topic.difficulty] = (counts[topic.difficulty] || 0) + 1;
            return counts;
        }, {});
    }
}

// Initialize the application
let aiHub;
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing AIKnowledgeHub');
    aiHub = new AIKnowledgeHub();
    window.aiHub = aiHub;
});