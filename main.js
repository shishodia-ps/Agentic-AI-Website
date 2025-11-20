// Agentic AI Knowledge Hub - Main JavaScript with Comprehensive Topics
class AIKnowledgeHub {
    constructor() {
        this.topics = this.loadTopics();
        this.init();
    }

    init() {
        this.initializeAnimations();
        this.setupEventListeners();
        this.renderTopicGrid();
        this.renderFeaturedTopics();
        this.animateStatistics();
        this.initializeSearch();
        this.createBackgroundPattern();
    }

    // Comprehensive topic database based on Microsoft AI curriculum and AI Engineering Toolkit
    loadTopics() {
        const defaultTopics = [
            // Generative AI Fundamentals
            {
                id: 1,
                title: "Introduction to Generative AI",
                category: "fundamentals",
                difficulty: "beginner",
                description: "Learn the fundamentals of generative AI, how Large Language Models work, and their core capabilities in content generation.",
                concepts: ["Generative AI", "LLMs", "Content Generation", "AI Fundamentals"],
                featured: true,
                progress: 0,
                content: `
                    <h2>What is Generative AI?</h2>
                    <p>Generative AI refers to artificial intelligence systems that can create new content, including text, images, code, and more. Unlike traditional AI that analyzes or classifies existing data, generative AI produces original outputs based on learned patterns.</p>
                    
                    <h3>Key Characteristics:</h3>
                    <ul>
                        <li><strong>Content Creation:</strong> Generates new, original content</li>
                        <li><strong>Pattern Learning:</strong> Learns from vast datasets</li>
                        <li><strong>Context Understanding:</strong> Maintains context in conversations</li>
                        <li><strong>Multi-modal:</strong> Can work with text, images, code, etc.</li>
                    </ul>
                    
                    <h3>How LLMs Work:</h3>
                    <p>Large Language Models (LLMs) are trained on massive amounts of text data to understand and generate human-like language. They use transformer architecture to process and predict text sequences.</p>
                    
                    <div class="code-block">
                    <pre><code># Simple example of using OpenAI API
import openai

# Initialize the client
client = openai.OpenAI(api_key="your-api-key")

# Generate text
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Explain generative AI in simple terms"}
    ]
)

print(response.choices[0].message.content)</code></pre>
                    </div>
                    
                    <h3>Practical Applications:</h3>
                    <ul>
                        <li>Content creation and writing assistance</li>
                        <li>Code generation and debugging</li>
                        <li>Customer service chatbots</li>
                        <li>Educational tutoring</li>
                        <li>Creative writing and brainstorming</li>
                    </ul>
                `
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
                content: `
                    <h2>Understanding Different LLMs</h2>
                    <p>The landscape of Large Language Models is diverse, with various models offering different capabilities, strengths, and use cases. Understanding these differences is crucial for selecting the right model for your application.</p>
                    
                    <h3>Major Model Categories:</h3>
                    
                    <h4>1. OpenAI Models (GPT Series)</h4>
                    <ul>
                        <li><strong>GPT-4:</strong> Most capable, best for complex reasoning</li>
                        <li><strong>GPT-3.5 Turbo:</strong> Fast, cost-effective for simpler tasks</li>
                        <li><strong>GPT-4 Turbo:</strong> Updated knowledge, larger context window</li>
                    </ul>
                    
                    <h4>2. Anthropic Models (Claude Series)</h4>
                    <ul>
                        <li><strong>Claude 3 Opus:</strong> Excellent reasoning and creativity</li>
                        <li><strong>Claude 3 Sonnet:</strong> Balanced performance and speed</li>
                        <li><strong>Claude 3 Haiku:</strong> Fastest, most cost-effective</li>
                    </ul>
                    
                    <h4>3. Open Source Models</h4>
                    <ul>
                        <li><strong>Llama 2/3:</strong> Meta's open-source models</li>
                        <li><strong>Mistral:</strong> Efficient, European-developed</li>
                        <li><strong>Gemma:</strong> Google's lightweight models</li>
                    </ul>
                    
                    <h3>Model Selection Criteria:</h3>
                    <div class="code-block">
                    <pre><code># Example: Comparing different models
import openai

models_to_test = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
prompt = "Write a short story about AI in exactly 100 words"

for model in models_to_test:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    
    print(f"Model: {model}")
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")
    print("-" * 50)</code></pre>
                    </div>
                    
                    <h3>Key Selection Factors:</h3>
                    <ul>
                        <li><strong>Task Complexity:</strong> Simple vs. complex reasoning</li>
                        <li><strong>Cost Considerations:</strong> Budget constraints and token usage</li>
                        <li><strong>Speed Requirements:</strong> Real-time vs. batch processing</li>
                        <li><strong>Context Length:</strong> Amount of information to process</li>
                        <li><strong>Accuracy Needs:</strong> Tolerance for errors or hallucinations</li>
                        <li><strong>Data Privacy:</strong> Open vs. closed source considerations</li>
                    </ul>
                `
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
                content: `
                    <h2>Responsible AI Principles</h2>
                    <p>As AI becomes more powerful and prevalent, it's crucial to develop and deploy AI systems responsibly. This involves considering the ethical implications, potential biases, and societal impacts of our AI applications.</p>
                    
                    <h3>Core Principles of Responsible AI:</h3>
                    
                    <h4>1. Fairness</h4>
                    <p>AI systems should treat all individuals and groups fairly, without discrimination based on protected characteristics like race, gender, or age.</p>
                    
                    <h4>2. Reliability & Safety</h4>
                    <p>AI systems should perform reliably and safely under normal and unexpected conditions, with appropriate fallback mechanisms.</p>
                    
                    <h4>3. Privacy & Security</h4>
                    <p>AI systems should be secure and respect privacy, protecting personal and sensitive data appropriately.</p>
                    
                    <h4>4. Inclusiveness</h4>
                    <p>AI systems should empower everyone and engage people in meaningful ways, considering diverse perspectives and needs.</p>
                    
                    <h4>5. Transparency</h4>
                    <p>AI systems should be understandable and interpretable, with clear documentation of their capabilities and limitations.</p>
                    
                    <h4>6. Accountability</h4>
                    <p>People should be responsible for AI systems, with appropriate human oversight and control mechanisms.</p>
                    
                    <h3>Practical Implementation:</h3>
                    <div class="code-block">
                    <pre><code># Example: Implementing content safety checks
import openai

def moderate_content(text):
    """Check content for safety violations"""
    response = client.moderations.create(input=text)
    
    results = response.results[0]
    if results.flagged:
        categories = [cat for cat, flagged in results.categories.items() if flagged]
        return False, categories
    
    return True, []

# Usage in application
def safe_ai_response(user_input):
    # First, check if input is safe
    is_safe, violations = moderate_content(user_input)
    
    if not is_safe:
        return "I cannot process that request due to safety concerns."
    
    # If safe, proceed with AI response
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_input}]
    )
    
    return response.choices[0].message.content</code></pre>
                    </div>
                    
                    <h3>Best Practices:</h3>
                    <ul>
                        <li>Implement content moderation and safety checks</li>
                        <li>Regular bias testing across different demographic groups</li>
                        <li>Clear documentation of model capabilities and limitations</li>
                        <li>Human oversight for critical decisions</li>
                        <li>Transparent data usage and privacy policies</li>
                        <li>Regular security audits and updates</li>
                    </ul>
                `
            },
            // Prompt Engineering
            {
                id: 4,
                title: "Prompt Engineering Fundamentals",
                category: "implementation",
                difficulty: "beginner",
                description: "Master the art and science of crafting effective prompts to guide AI models toward desired outputs and behaviors.",
                concepts: ["Prompt Engineering", "Zero-shot", "Few-shot", "Chain-of-Thought", "System Prompts"],
                featured: true,
                progress: 0,
                content: `
                    <h2>The Art of Prompt Engineering</h2>
                    <p>Prompt engineering is the practice of designing and optimizing inputs to AI models to achieve desired outputs. It's both an art and a science that can dramatically improve the quality and consistency of AI responses.</p>
                    
                    <h3>Core Components of a Prompt:</h3>
                    <ul>
                        <li><strong>Context:</strong> Background information and setting</li>
                        <li><strong>Instruction:</strong> Clear directive on what to do</li>
                        <li><strong>Input Data:</strong> The specific content to process</li>
                        <li><strong>Output Format:</strong> Desired structure of the response</li>
                        <li><strong>Examples:</strong> Sample inputs and outputs (for few-shot)</li>
                    </ul>
                    
                    <h3>Prompting Techniques:</h3>
                    
                    <h4>1. Zero-Shot Prompting</h4>
                    <p>Providing instructions without examples:</p>
                    <div class="code-block">
                    <pre><code>prompt = """
Classify the following text as positive, negative, or neutral sentiment:

Text: "I love this new AI technology!"
Sentiment: """</code></pre>
                    </div>
                    
                    <h4>2. Few-Shot Prompting</h4>
                    <p>Providing examples to guide the model:</p>
                    <div class="code-block">
                    <pre><code>prompt = """
Classify the sentiment of these texts:

Example 1:
Text: "This product is amazing!"
Sentiment: Positive

Example 2:
Text: "I'm not satisfied with the service"
Sentiment: Negative

Now classify:
Text: "The weather is okay today"
Sentiment: """</code></pre>
                    </div>
                    
                    <h4>3. Chain-of-Thought (CoT)</h4>
                    <p>Encouraging step-by-step reasoning:</p>
                    <div class="code-block">
                    <pre><code>prompt = """
Solve this math problem step by step:

Problem: If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire trip?

Let's think step by step:
1. Total distance = 120 + 180 = 300 miles
2. Total time = 2 + 3 = 5 hours
3. Average speed = Total distance / Total time = 300 / 5 = 60 mph

Answer: 60 mph"""</code></pre>
                    </div>
                    
                    <h3>Advanced Techniques:</h3>
                    <ul>
                        <li><strong>Role Playing:</strong> Assign a specific role to the AI</li>
                        <li><strong>Delimiters:</strong> Use special characters to separate sections</li>
                        <li><strong>Output Formatting:</strong> Specify exact format requirements</li>
                        <li><strong>Temperature Control:</strong> Adjust creativity vs. determinism</li>
                        <li><strong>Iterative Refinement:</strong> Improve prompts based on outputs</li>
                    </ul>
                    
                    <div class="code-block">
                    <pre><code># Practical implementation
import openai

def generate_with_prompt(prompt, temperature=0.7):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=500
    )
    return response.choices[0].message.content

# Example usage
user_prompt = """
Act as a Python programming expert. Explain the concept of decorators
in simple terms with a practical example. Use analogies and keep it
under 200 words.
"""

result = generate_with_prompt(user_prompt, temperature=0.3)
print(result)</code></pre>
                    </div>
                `
            },
            // RAG and Vector Databases
            {
                id: 5,
                title: "Retrieval-Augmented Generation (RAG)",
                category: "implementation",
                difficulty: "intermediate",
                description: "Learn how to enhance LLM responses with external knowledge using RAG techniques and vector databases.",
                concepts: ["RAG", "Vector Databases", "Embeddings", "Semantic Search", "Knowledge Retrieval"],
                featured: true,
                progress: 0,
                content: `
                    <h2>Retrieval-Augmented Generation (RAG)</h2>
                    <p>RAG is a technique that enhances Large Language Models by providing them with external knowledge from a retrieval system. This allows the model to access up-to-date information and reduce hallucinations.</p>
                    
                    <h3>How RAG Works:</h3>
                    <ol>
                        <li><strong>Document Ingestion:</strong> Process and index external documents</li>
                        <li><strong>Query Embedding:</strong> Convert user query to vector representation</li>
                        <li><strong>Semantic Search:</strong> Find relevant documents in the vector database</li>
                        <li><strong>Context Augmentation:</strong> Add retrieved context to the prompt</li>
                        <li><strong>Response Generation:</strong> Generate answer using augmented context</li>
                    </ol>
                    
                    <h3>Vector Database Options:</h3>
                    <ul>
                        <li><strong>Pinecone:</strong> Managed vector database service</li>
                        <li><strong>Chroma:</strong> Open-source embedding database</li>
                        <li><strong>Weaviate:</strong> Vector search engine with GraphQL API</li>
                        <li><strong>Qdrant:</strong> Vector similarity search engine</li>
                        <li><strong>FAISS:</strong> Facebook's library for efficient similarity search</li>
                    </ul>
                    
                    <div class="code-block">
                    <pre><code># Simple RAG implementation with ChromaDB
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai

class SimpleRAG:
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings())
        self.collection = self.client.create_collection("knowledge_base")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_documents(self, documents):
        """Add documents to the knowledge base"""
        for i, doc in enumerate(documents):
            embedding = self.embedding_model.encode([doc])[0]
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[doc],
                ids=[f"doc_{i}"]
            )
    
    def query(self, question, n_results=3):
        """Query the knowledge base"""
        # Get query embedding
        query_embedding = self.embedding_model.encode([question])[0]
        
        # Search for relevant documents
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Combine retrieved documents
        context = " ".join(results['documents'][0])
        
        # Generate response with context
        prompt = f"""
        Based on the following context, answer the question:
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

# Usage example
rag = SimpleRAG()

# Add documents to knowledge base
documents = [
    "Python is a high-level programming language created by Guido van Rossum.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "RAG combines retrieval systems with generative AI to provide accurate responses."
]

rag.add_documents(documents)

# Query the system
answer = rag.query("What is RAG?")
print(answer)</code></pre>
                    </div>
                    
                    <h3>RAG Best Practices:</h3>
                    <ul>
                        <li><strong>Chunk Size:</strong> Balance between context and precision</li>
                        <li><strong>Overlap:</strong> Include overlapping text between chunks</li>
                        <li><strong>Metadata:</strong> Add relevant metadata for filtering</li>
                        <li><strong>Re-ranking:</strong> Use multiple retrieval strategies</li>
                        <li><strong>Hybrid Search:</strong> Combine semantic and keyword search</li>
                        <li><strong>Evaluation:</strong> Continuously test and improve retrieval quality</li>
                    </ul>
                `
            },
            // AI Agents
            {
                id: 6,
                title: "Building AI Agents",
                category: "architecture",
                difficulty: "intermediate",
                description: "Create autonomous AI agents that can perceive, reason, act, and learn to accomplish complex tasks.",
                concepts: ["AI Agents", "Autonomous Systems", "Tool Use", "Planning", "Multi-Agent Systems"],
                featured: true,
                progress: 0,
                content: `
                    <h2>Understanding AI Agents</h2>
                    <p>AI agents are autonomous entities that perceive their environment, make decisions, and take actions to achieve specific goals. They represent the next evolution beyond simple question-answering systems.</p>
                    
                    <h3>Agent Components:</h3>
                    <ul>
                        <li><strong>Perception:</strong> Sensing and understanding the environment</li>
                        <li><strong>Reasoning:</strong> Processing information and making decisions</li>
                        <li><strong>Planning:</strong> Creating strategies to achieve goals</li>
                        <li><strong>Action:</strong> Executing planned actions</li>
                        <li><strong>Learning:</strong> Improving performance over time</li>
                        <li><strong>Memory:</strong> Storing and retrieving information</li>
                    </ul>
                    
                    <h3>Agent Design Patterns:</h3>
                    
                    <h4>1. ReAct (Reasoning and Acting)</h4>
                    <p>Interleaves reasoning and action, allowing the agent to think about its next steps:</p>
                    <div class="code-block">
                    <pre><code>class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
    
    def act(self, query):
        thought_process = []
        
        while True:
            # Generate thought and action
            prompt = f"""
            Thought: {thought_process[-1] if thought_process else 'I need to solve this query'}
            Action: """
            
            response = self.llm.generate(prompt)
            
            if "Action:" in response:
                action = response.split("Action:")[1].strip()
                
                if action.startswith("Final Answer:"):
                    return action.split("Final Answer:")[1].strip()
                
                # Execute tool if needed
                for tool_name, tool in self.tools.items():
                    if action.startswith(tool_name):
                        result = tool.execute(action)
                        thought_process.append(f"Observation: {result}")
            
            thought_process.append(response)</code></pre>
                    </div>
                    
                    <h4>2. Planning Agents</h4>
                    <p>Create detailed plans before executing actions:</p>
                    <div class="code-block">
                    <pre><code>class PlanningAgent:
    def __init__(self, llm):
        self.llm = llm
    
    def solve_task(self, task):
        # Create plan
        plan_prompt = f"""
        Create a detailed step-by-step plan to accomplish: {task}
        
        Plan:
        1. """
        
        plan = self.llm.generate(plan_prompt)
        steps = plan.split('\n')
        
        # Execute plan
        results = []
        for step in steps:
            if step.strip():
                result = self.execute_step(step)
                results.append(result)
        
        return results
    
    def execute_step(self, step):
        # Execute individual step
        execution_prompt = f"""
        Execute this step: {step}
        
        Result:"""
        
        return self.llm.generate(execution_prompt)</code></pre>
                    </div>
                    
                    <h3>Tool Integration:</h3>
                    <p>Agents become powerful when they can use external tools:</p>
                    <div class="code-block">
                    <pre><code>class Tool:
    def __init__(self, name, description, function):
        self.name = name
        self.description = description
        self.function = function
    
    def execute(self, parameters):
        return self.function(parameters)

# Define tools
calculator = Tool(
    name="calculate",
    description="Perform mathematical calculations",
    function=lambda expr: eval(expr)
)

web_search = Tool(
    name="search",
    description="Search the web for information",
    function=lambda query: f"Search results for: {query}"
)

file_system = Tool(
    name="read_file",
    description="Read contents of a file",
    function=lambda filename: f"Contents of {filename}"
)

# Agent with tool access
class ToolAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
    
    def process_query(self, query):
        # Agent decides which tool to use
        tool_selection_prompt = f"""
        Query: {query}
        
        Available tools:
        {chr(10).join([f"- {tool.name}: {tool.description}" for tool in self.tools])}
        
        Which tool should I use? Answer with just the tool name.
        """
        
        selected_tool = self.llm.generate(tool_selection_prompt).strip()
        
        # Use the selected tool
        for tool in self.tools:
            if tool.name == selected_tool:
                parameters = self.extract_parameters(query)
                return tool.execute(parameters)
        
        return "No suitable tool found"
    
    def extract_parameters(self, query):
        # Extract parameters from query
        param_prompt = f"""
        Extract the parameters from this query: {query}
        
        Parameters:"""
        
        return self.llm.generate(param_prompt).strip()</code></pre>
                    </div>
                    
                    <h3>Multi-Agent Systems:</h3>
                    <p>Multiple agents working together can solve complex problems:</p>
                    <ul>
                        <li><strong>Specialized Agents:</strong> Each agent has specific expertise</li>
                        <li><strong>Coordinator Agent:</strong> Manages task distribution</li>
                        <li><strong>Communication:</strong> Agents share information and results</li>
                        <li><strong>Consensus:</strong> Agents agree on final solutions</li>
                    </ul>
                `
            },
            // LangChain Framework
            {
                id: 7,
                title: "LangChain Framework",
                category: "frameworks",
                difficulty: "intermediate",
                description: "Master LangChain for building applications with language models using chains, agents, and memory systems.",
                concepts: ["LangChain", "Chains", "Agents", "Memory", "Tools", "Prompts"],
                featured: true,
                progress: 0,
                content: `
                    <h2>LangChain Framework</h2>
                    <p>LangChain is a comprehensive framework for developing applications powered by language models. It provides modular components for chains, agents, memory, and tools that can be combined to create sophisticated AI applications.</p>
                    
                    <h3>Core Components:</h3>
                    
                    <h4>1. Models</h4>
                    <p>LangChain supports various language models and provides a unified interface:</p>
                    <div class="code-block">
                    <pre><code>from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# Chat models (optimized for conversation)
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key="your-api-key"
)

# Text completion models
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.7,
    openai_api_key="your-api-key"
)

# Generate responses
response = chat_model.invoke("Hello, how are you?")
print(response.content)</code></pre>
                    </div>
                    
                    <h4>2. Prompts</h4>
                    <p>Structured way to create and manage prompts:</p>
                    <div class="code-block">
                    <pre><code>from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Chat prompt template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specializing in {topic}."),
    ("human", "Please explain {concept} in simple terms."),
    ("ai", "I'd be happy to explain {concept} to you."),
    ("human", "What are the key benefits?")
])

# Fill in the template
prompt = chat_template.format_messages(
    topic="machine learning",
    concept="neural networks"
)

# Regular prompt template
template = """
You are a {role} expert.
Please provide {num_points} key points about {topic}.
Make it accessible for {audience}.
"""

prompt_template = PromptTemplate(
    input_variables=["role", "num_points", "topic", "audience"],
    template=template
)

formatted_prompt = prompt_template.format(
    role="data science",
    num_points=5,
    topic="data visualization",
    audience="beginners"
)</code></pre>
                    </div>
                    
                    <h4>3. Chains</h4>
                    <p>Sequences of calls to components:</p>
                    <div class="code-block">
                    <pre><code>from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate

# Create individual chains
# Chain 1: Generate a topic
topic_template = "Generate an interesting topic about {field}"
topic_prompt = PromptTemplate(
    input_variables=["field"],
    template=topic_template
)
topic_chain = LLMChain(llm=llm, prompt=topic_prompt)

# Chain 2: Write about the topic
writing_template = "Write a 100-word explanation about {topic} for beginners"
writing_prompt = PromptTemplate(
    input_variables=["topic"],
    template=writing_template
)
writing_chain = LLMChain(llm=llm, prompt=writing_prompt)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[topic_chain, writing_chain],
    verbose=True
)

# Run the combined chain
result = overall_chain.run("artificial intelligence")
print(result)</code></pre>
                    </div>
                    
                    <h4>4. Memory</h4>
                    <p>Persist information across interactions:</p>
                    <div class="code-block">
                    <pre><code>from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain

# Simple conversation memory
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=chat_model,
    memory=memory,
    verbose=True
)

# Have a conversation
response1 = conversation.predict(input="Hi, my name is Alice")
print(response1)

response2 = conversation.predict(input="What's my name?")
print(response2)  # Should remember the name

# Summary memory for longer conversations
summary_memory = ConversationSummaryMemory(llm=llm)
summary_conversation = ConversationChain(
    llm=chat_model,
    memory=summary_memory,
    verbose=True
)</code></pre>
                    </div>
                    
                    <h4>5. Tools and Agents</h4>
                    <p>Give models access to external capabilities:</p>
                    <div class="code-block">
                    <pre><code>from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Define custom tools
def calculate(expression):
    """Perform mathematical calculations"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

def get_weather(location):
    """Get weather information for a location"""
    # In practice, this would call a weather API
    return f"The weather in {location} is sunny and 75°F"

# Create tools
calculator_tool = Tool(
    name="Calculator",
    func=calculate,
    description="Useful for mathematical calculations"
)

weather_tool = Tool(
    name="Weather",
    func=get_weather,
    description="Get weather information for a location"
)

# Initialize agent with tools
tools = [calculator_tool, weather_tool]

agent = initialize_agent(
    tools,
    chat_model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
result = agent.run("What's the weather in New York and what's 15 * 23?")
print(result)</code></pre>
                    </div>
                    
                    <h3>Best Practices:</h3>
                    <ul>
                        <li><strong>Error Handling:</strong> Implement robust error handling and fallbacks</li>
                        <li><strong>Cost Management:</strong> Monitor token usage and implement caching</li>
                        <li><strong>Async Operations:</strong> Use async chains for better performance</li>
                        <li><strong>Testing:</strong> Test chains and agents with various inputs</li>
                        <li><strong>Monitoring:</strong> Use LangSmith for debugging and monitoring</li>
                        <li><strong>Security:</strong> Never expose API keys in client-side code</li>
                    </ul>
                `
            }
        ];

        // Load from localStorage or use defaults
        const savedTopics = localStorage.getItem('ai-topics');
        return savedTopics ? JSON.parse(savedTopics) : defaultTopics;
    }

    saveTopics() {
        localStorage.setItem('ai-topics', JSON.stringify(this.topics));
    }

    initializeAnimations() {
        // Typewriter effect for hero title
        if (document.getElementById('typed-text')) {
            new Typed('#typed-text', {
                strings: [
                    'Master Agentic AI',
                    'Build Intelligent Systems',
                    'Explore AI Frameworks',
                    'Learn Autonomous Agents'
                ],
                typeSpeed: 60,
                backSpeed: 40,
                backDelay: 2000,
                loop: true,
                showCursor: true,
                cursorChar: '|'
            });
        }

        // Initialize scroll animations
        this.observeElements();
    }

    observeElements() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animated');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        document.querySelectorAll('.fade-in, .stagger-item').forEach(el => {
            observer.observe(el);
        });
    }

    setupEventListeners() {
        // Search functionality
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => this.handleSearch(e.target.value));
            searchInput.addEventListener('focus', () => this.showSearchResults());
            searchInput.addEventListener('blur', () => {
                setTimeout(() => this.hideSearchResults(), 200);
            });
        }

        // Modal functionality
        const addTopicBtn = document.getElementById('addTopicBtn');
        const addTopicModal = document.getElementById('addTopicModal');
        const closeModal = document.getElementById('closeModal');
        const cancelAdd = document.getElementById('cancelAdd');
        const addTopicForm = document.getElementById('addTopicForm');

        if (addTopicBtn && addTopicModal) {
            addTopicBtn.addEventListener('click', () => this.showModal());
            closeModal?.addEventListener('click', () => this.hideModal());
            cancelAdd?.addEventListener('click', () => this.hideModal());
            
            // Close modal on outside click
            addTopicModal.addEventListener('click', (e) => {
                if (e.target === addTopicModal) {
                    this.hideModal();
                }
            });
        }

        if (addTopicForm) {
            addTopicForm.addEventListener('submit', (e) => this.handleAddTopic(e));
        }

        // Mobile menu
        const mobileMenuBtn = document.getElementById('mobileMenuBtn');
        if (mobileMenuBtn) {
            mobileMenuBtn.addEventListener('click', () => this.toggleMobileMenu());
        }
    }

    renderTopicGrid() {
        const topicGrid = document.getElementById('topicGrid');
        if (!topicGrid) return;

        const categories = this.groupTopicsByCategory();
        topicGrid.innerHTML = '';

        Object.entries(categories).forEach(([category, topics]) => {
            const categoryCard = this.createCategoryCard(category, topics);
            topicGrid.appendChild(categoryCard);
        });

        // Animate cards
        anime({
            targets: '.topic-card',
            opacity: [0, 1],
            translateY: [30, 0],
            delay: anime.stagger(100),
            duration: 600,
            easing: 'easeOutQuart'
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
                <button class="text-copper-accent hover:text-sage-deep text-sm font-semibold" onclick="this.viewTopic(${topic.id})">
                    Learn More →
                </button>
            </div>
        `;

        card.addEventListener('click', () => this.viewTopic(topic.id));
        return card;
    }

    animateStatistics() {
        const stats = this.calculateStatistics();
        
        // Animate counters
        this.animateCounter('topicsCount', stats.totalTopics);
        this.animateCounter('categoriesCount', stats.totalCategories);
        this.animateCounter('frameworksCount', stats.frameworkTopics);
        this.animateCounter('advancedCount', stats.advancedTopics);

        // Animate progress rings
        setTimeout(() => {
            this.animateProgressRing('topicsProgress', 100);
            this.animateProgressRing('categoriesProgress', 100);
            this.animateProgressRing('frameworksProgress', (stats.frameworkTopics / stats.totalTopics) * 100);
            this.animateProgressRing('advancedProgress', (stats.advancedTopics / stats.totalTopics) * 100);
        }, 500);
    }

    animateCounter(elementId, targetValue) {
        const element = document.getElementById(elementId);
        if (!element) return;

        anime({
            targets: { value: 0 },
            value: targetValue,
            duration: 2000,
            easing: 'easeOutQuart',
            update: function(anim) {
                element.textContent = Math.round(anim.animatables[0].target.value);
            }
        });
    }

    animateProgressRing(elementId, percentage) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const circumference = 2 * Math.PI * 45; // radius = 45
        const offset = circumference - (percentage / 100) * circumference;
        
        anime({
            targets: element,
            strokeDashoffset: [circumference, offset],
            duration: 1500,
            easing: 'easeOutQuart'
        });
    }

    initializeSearch() {
        // This will be implemented with the search functionality
        this.searchIndex = this.topics.map(topic => ({
            ...topic,
            searchText: `${topic.title} ${topic.description} ${topic.concepts.join(' ')} ${topic.category}`.toLowerCase()
        }));
    }

    handleSearch(query) {
        if (!query.trim()) {
            this.hideSearchResults();
            return;
        }

        const results = this.searchIndex.filter(topic => 
            topic.searchText.includes(query.toLowerCase())
        );

        this.displaySearchResults(results);
    }

    displaySearchResults(results) {
        const searchResults = document.getElementById('searchResults');
        if (!searchResults) return;

        if (results.length === 0) {
            searchResults.innerHTML = '<div class="search-result-item text-gray-500">No topics found</div>';
        } else {
            searchResults.innerHTML = results.map(topic => `
                <div class="search-result-item" onclick="window.aiHub.viewTopic(${topic.id})">
                    <div class="font-semibold text-charcoal-warm">${topic.title}</div>
                    <div class="text-sm text-gray-600">${topic.category} • ${topic.difficulty}</div>
                </div>
            `).join('');
        }

        searchResults.style.display = 'block';
    }

    showSearchResults() {
        const searchResults = document.getElementById('searchResults');
        const searchInput = document.getElementById('searchInput');
        if (searchResults && searchInput && searchInput.value.trim()) {
            searchResults.style.display = 'block';
        }
    }

    hideSearchResults() {
        const searchResults = document.getElementById('searchResults');
        if (searchResults) {
            searchResults.style.display = 'none';
        }
    }

    showModal() {
        const modal = document.getElementById('addTopicModal');
        if (modal) {
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    }

    hideModal() {
        const modal = document.getElementById('addTopicModal');
        if (modal) {
            modal.classList.remove('active');
            document.body.style.overflow = '';
            this.resetForm();
        }
    }

    resetForm() {
        const form = document.getElementById('addTopicForm');
        if (form) {
            form.reset();
        }
    }

    handleAddTopic(e) {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const newTopic = {
            id: Date.now(),
            title: document.getElementById('topicTitle').value,
            category: document.getElementById('topicCategory').value,
            difficulty: document.getElementById('topicDifficulty').value,
            description: document.getElementById('topicDescription').value,
            concepts: document.getElementById('topicConcepts').value.split(',').map(c => c.trim()).filter(c => c),
            featured: false,
            progress: 0,
            content: "Content to be added..."
        };

        this.topics.push(newTopic);
        this.saveTopics();
        this.initializeSearch();
        
        // Refresh displays
        this.renderTopicGrid();
        this.renderFeaturedTopics();
        this.animateStatistics();
        
        this.hideModal();
        
        // Show success message
        this.showNotification('Topic added successfully!', 'success');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 z-50 p-4 rounded-lg text-white font-semibold transform translate-x-full transition-transform duration-300`;
        notification.style.background = type === 'success' ? 'var(--success-green)' : 'var(--copper-accent)';
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    viewTopic(topicId) {
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

    calculateStatistics() {
        const categories = this.groupTopicsByCategory();
        return {
            totalTopics: this.topics.length,
            totalCategories: Object.keys(categories).length,
            frameworkTopics: this.topics.filter(t => t.category === 'frameworks').length,
            advancedTopics: this.topics.filter(t => t.difficulty === 'advanced').length
        };
    }

    createBackgroundPattern() {
        const bgPattern = document.getElementById('bgPattern');
        if (!bgPattern) return;

        // Create p5.js sketch for background pattern
        const sketch = (p) => {
            let nodes = [];
            
            p.setup = () => {
                const canvas = p.createCanvas(p.windowWidth, p.windowHeight);
                canvas.parent('bgPattern');
                
                // Create random nodes
                for (let i = 0; i < 50; i++) {
                    nodes.push({
                        x: p.random(p.width),
                        y: p.random(p.height),
                        vx: p.random(-0.5, 0.5),
                        vy: p.random(-0.5, 0.5)
                    });
                }
            };
            
            p.draw = () => {
                p.clear();
                p.stroke(45, 90, 61, 30);
                p.strokeWeight(1);
                p.noFill();
                
                // Update and draw nodes
                nodes.forEach((node, i) => {
                    node.x += node.vx;
                    node.y += node.vy;
                    
                    // Wrap around edges
                    if (node.x < 0) node.x = p.width;
                    if (node.x > p.width) node.x = 0;
                    if (node.y < 0) node.y = p.height;
                    if (node.y > p.height) node.y = 0;
                    
                    // Draw connections to nearby nodes
                    nodes.forEach((other, j) => {
                        if (i !== j) {
                            const distance = p.dist(node.x, node.y, other.x, other.y);
                            if (distance < 100) {
                                p.line(node.x, node.y, other.x, other.y);
                            }
                        }
                    });
                    
                    // Draw node
                    p.fill(45, 90, 61, 50);
                    p.noStroke();
                    p.circle(node.x, node.y, 3);
                });
            };
            
            p.windowResized = () => {
                p.resizeCanvas(p.windowWidth, p.windowHeight);
            };
        };
        
        new p5(sketch);
    }

    toggleMobileMenu() {
        // Mobile menu implementation
        console.log('Mobile menu toggled');
    }
}

// Initialize the application
let aiHub;
document.addEventListener('DOMContentLoaded', () => {
    aiHub = new AIKnowledgeHub();
});

// Global functions for HTML onclick handlers
window.viewTopic = (topicId) => aiHub?.viewTopic(topicId);
window.aiHub = aiHub;