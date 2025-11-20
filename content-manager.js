/**
 * AI Engineering Knowledge Hub - Content Management System
 * Advanced visual editor with CMS capabilities
 */

class AIKnowledgeCMS {
    constructor() {
        this.pages = new Map();
        this.templates = new Map();
        this.media = new Map();
        this.users = new Map();
        this.settings = {
            siteName: 'AI Engineering Knowledge Hub',
            theme: 'default',
            colorScheme: 'blue',
            fontFamily: 'Inter',
            autoSave: true,
            autoSaveInterval: 30000
        };
        
        this.currentUser = {
            id: 'admin',
            name: 'Administrator',
            role: 'admin',
            permissions: ['read', 'write', 'delete', 'publish']
        };
        
        this.init();
    }

    init() {
        this.loadSettings();
        this.loadPages();
        this.setupAutoSave();
        this.initializeDefaultPages();
        console.log('AI Knowledge CMS initialized');
    }

    // Settings Management
    loadSettings() {
        const savedSettings = localStorage.getItem('cms_settings');
        if (savedSettings) {
            this.settings = { ...this.settings, ...JSON.parse(savedSettings) };
        }
    }

    saveSettings() {
        localStorage.setItem('cms_settings', JSON.stringify(this.settings));
    }

    updateSetting(key, value) {
        this.settings[key] = value;
        this.saveSettings();
        this.applySettings();
    }

    applySettings() {
        // Apply theme settings
        document.documentElement.style.setProperty('--primary-color', this.settings.colorScheme);
        document.body.style.fontFamily = this.settings.fontFamily;
    }

    // Page Management
    loadPages() {
        const savedPages = localStorage.getItem('cms_pages');
        if (savedPages) {
            const pagesData = JSON.parse(savedPages);
            this.pages = new Map(Object.entries(pagesData));
        }
    }

    savePages() {
        const pagesData = Object.fromEntries(this.pages);
        localStorage.setItem('cms_pages', JSON.stringify(pagesData));
    }

    createPage(pageData) {
        const page = {
            id: pageData.id || 'page_' + Date.now(),
            title: pageData.title || 'New Page',
            url: pageData.url || 'new-page.html',
            description: pageData.description || '',
            content: pageData.content || '',
            sections: pageData.sections || [],
            boxes: pageData.boxes || [],
            metadata: {
                author: this.currentUser.id,
                created: new Date().toISOString(),
                modified: new Date().toISOString(),
                version: 1,
                status: 'draft',
                tags: pageData.tags || [],
                category: pageData.category || 'general'
            },
            settings: {
                layout: pageData.layout || 'standard',
                sidebar: pageData.sidebar !== false,
                toc: pageData.toc !== false,
                comments: pageData.comments !== false
            }
        };

        this.pages.set(page.id, page);
        this.savePages();
        this.logAction('create', 'page', page.id);
        
        return page;
    }

    updatePage(pageId, updates) {
        const page = this.pages.get(pageId);
        if (!page) {
            throw new Error(`Page ${pageId} not found`);
        }

        // Update page data
        Object.assign(page, updates);
        page.metadata.modified = new Date().toISOString();
        page.metadata.version += 1;

        this.pages.set(pageId, page);
        this.savePages();
        this.logAction('update', 'page', pageId);
        
        return page;
    }

    deletePage(pageId) {
        if (!this.pages.has(pageId)) {
            throw new Error(`Page ${pageId} not found`);
        }

        this.pages.delete(pageId);
        this.savePages();
        this.logAction('delete', 'page', pageId);
        
        return true;
    }

    getPage(pageId) {
        return this.pages.get(pageId);
    }

    getAllPages() {
        return Array.from(this.pages.values());
    }

    searchPages(query) {
        const results = [];
        const searchTerm = query.toLowerCase();
        
        for (const page of this.pages.values()) {
            if (
                page.title.toLowerCase().includes(searchTerm) ||
                page.description.toLowerCase().includes(searchTerm) ||
                page.content.toLowerCase().includes(searchTerm) ||
                page.metadata.tags.some(tag => tag.toLowerCase().includes(searchTerm))
            ) {
                results.push(page);
            }
        }
        
        return results;
    }

    // Content Block Management
    addContentBlock(pageId, blockData) {
        const page = this.getPage(pageId);
        if (!page) {
            throw new Error(`Page ${pageId} not found`);
        }

        const block = {
            id: blockData.id || 'block_' + Date.now(),
            type: blockData.type || 'text',
            content: blockData.content || '',
            position: blockData.position || page.boxes.length,
            settings: {
                visible: blockData.visible !== false,
                collapsible: blockData.collapsible || false,
                styled: blockData.styled !== false
            },
            metadata: {
                created: new Date().toISOString(),
                modified: new Date().toISOString(),
                author: this.currentUser.id
            }
        };

        page.boxes.push(block);
        this.updatePage(pageId, { boxes: page.boxes });
        
        return block;
    }

    updateContentBlock(pageId, blockId, updates) {
        const page = this.getPage(pageId);
        if (!page) {
            throw new Error(`Page ${pageId} not found`);
        }

        const blockIndex = page.boxes.findIndex(block => block.id === blockId);
        if (blockIndex === -1) {
            throw new Error(`Block ${blockId} not found`);
        }

        Object.assign(page.boxes[blockIndex], updates);
        page.boxes[blockIndex].metadata.modified = new Date().toISOString();
        
        this.updatePage(pageId, { boxes: page.boxes });
        
        return page.boxes[blockIndex];
    }

    deleteContentBlock(pageId, blockId) {
        const page = this.getPage(pageId);
        if (!page) {
            throw new Error(`Page ${pageId} not found`);
        }

        const blockIndex = page.boxes.findIndex(block => block.id === blockId);
        if (blockIndex === -1) {
            throw new Error(`Block ${blockId} not found`);
        }

        page.boxes.splice(blockIndex, 1);
        this.updatePage(pageId, { boxes: page.boxes });
        
        return true;
    }

    // Template Management
    createTemplate(templateData) {
        const template = {
            id: templateData.id || 'template_' + Date.now(),
            name: templateData.name || 'New Template',
            description: templateData.description || '',
            content: templateData.content || '',
            type: templateData.type || 'page',
            category: templateData.category || 'general',
            settings: templateData.settings || {},
            metadata: {
                author: this.currentUser.id,
                created: new Date().toISOString(),
                modified: new Date().toISOString(),
                version: 1
            }
        };

        this.templates.set(template.id, template);
        this.saveTemplates();
        this.logAction('create', 'template', template.id);
        
        return template;
    }

    applyTemplate(pageId, templateId) {
        const page = this.getPage(pageId);
        const template = this.templates.get(templateId);
        
        if (!page || !template) {
            throw new Error('Page or template not found');
        }

        // Apply template content and settings
        if (template.content) {
            page.content = template.content;
        }
        
        if (template.settings) {
            page.settings = { ...page.settings, ...template.settings };
        }

        this.updatePage(pageId, page);
        this.logAction('apply', 'template', templateId, { pageId });
        
        return page;
    }

    saveTemplates() {
        const templatesData = Object.fromEntries(this.templates);
        localStorage.setItem('cms_templates', JSON.stringify(templatesData));
    }

    // Media Management
    uploadMedia(file, metadata = {}) {
        const mediaId = 'media_' + Date.now();
        const media = {
            id: mediaId,
            name: file.name,
            type: file.type,
            size: file.size,
            url: URL.createObjectURL(file), // In production, upload to cloud storage
            metadata: {
                ...metadata,
                uploaded: new Date().toISOString(),
                uploadedBy: this.currentUser.id
            }
        };

        this.media.set(mediaId, media);
        this.saveMedia();
        this.logAction('upload', 'media', mediaId);
        
        return media;
    }

    saveMedia() {
        const mediaData = Object.fromEntries(this.media);
        localStorage.setItem('cms_media', JSON.stringify(mediaData));
    }

    // Auto-save functionality
    setupAutoSave() {
        if (this.settings.autoSave) {
            setInterval(() => {
                this.autoSave();
            }, this.settings.autoSaveInterval);
        }
    }

    autoSave() {
        // Save all pending changes
        this.savePages();
        this.saveSettings();
        this.saveTemplates();
        this.saveMedia();
        
        console.log('Auto-save completed');
    }

    // Logging and Analytics
    logAction(action, resourceType, resourceId, metadata = {}) {
        const logEntry = {
            timestamp: new Date().toISOString(),
            user: this.currentUser.id,
            action,
            resourceType,
            resourceId,
            metadata,
            sessionId: this.sessionId || 'unknown'
        };

        // In production, send to analytics service
        console.log('CMS Action:', logEntry);
        
        // Store in localStorage for demo
        const logs = JSON.parse(localStorage.getItem('cms_logs') || '[]');
        logs.push(logEntry);
        
        // Keep only last 100 logs
        if (logs.length > 100) {
            logs.splice(0, logs.length - 100);
        }
        
        localStorage.setItem('cms_logs', JSON.stringify(logs));
    }

    // Utility Functions
    generateId(prefix = 'item') {
        return prefix + '_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Backup and Restore
    createBackup() {
        const backup = {
            timestamp: new Date().toISOString(),
            version: '1.0',
            pages: Object.fromEntries(this.pages),
            templates: Object.fromEntries(this.templates),
            media: Object.fromEntries(this.media),
            settings: this.settings,
            user: this.currentUser.id
        };

        const blob = new Blob([JSON.stringify(backup, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `cms_backup_${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
        this.logAction('backup', 'system', 'full');
        
        return backup;
    }

    restoreBackup(backupData) {
        try {
            if (backupData.pages) {
                this.pages = new Map(Object.entries(backupData.pages));
            }
            if (backupData.templates) {
                this.templates = new Map(Object.entries(backupData.templates));
            }
            if (backupData.media) {
                this.media = new Map(Object.entries(backupData.media));
            }
            if (backupData.settings) {
                this.settings = { ...this.settings, ...backupData.settings };
            }

            this.savePages();
            this.saveTemplates();
            this.saveMedia();
            this.saveSettings();
            
            this.logAction('restore', 'system', 'full');
            return true;
        } catch (error) {
            console.error('Backup restoration failed:', error);
            return false;
        }
    }

    // Initialize default content
    initializeDefaultPages() {
        if (this.pages.size === 0) {
            // Create default pages for the AI Engineering Knowledge Hub
            const defaultPages = [
                {
                    id: 'ai-stack-overview',
                    title: 'AI Engineering Stack Overview',
                    url: 'topic-ai-stack-overview.html',
                    description: 'Complete architecture of modern AI engineering systems',
                    category: 'foundations',
                    tags: ['architecture', 'overview', 'stack']
                },
                {
                    id: 'llm-foundations',
                    title: 'LLM Foundations',
                    url: 'topic-llm-foundations.html',
                    description: 'Large Language Models: architecture, training, and capabilities',
                    category: 'foundations',
                    tags: ['llm', 'models', 'architecture']
                },
                {
                    id: 'vector-databases',
                    title: 'Vector Databases',
                    url: 'topic-vector-databases.html',
                    description: 'Specialized databases for high-dimensional vector storage and retrieval',
                    category: 'infrastructure',
                    tags: ['databases', 'vectors', 'storage']
                },
                {
                    id: 'embedding-models',
                    title: 'Embedding Models',
                    url: 'topic-embedding-models.html',
                    description: 'Text and multimodal embedding generation and optimization',
                    category: 'models',
                    tags: ['embeddings', 'models', 'nlp']
                },
                {
                    id: 'prompt-engineering',
                    title: 'Prompt Engineering Framework',
                    url: 'topic-prompt-engineering.html',
                    description: 'Systematic approaches to prompt design and optimization',
                    category: 'techniques',
                    tags: ['prompts', 'engineering', 'optimization']
                },
                {
                    id: 'retrieval-systems',
                    title: 'Retrieval Systems',
                    url: 'topic-retrieval-systems.html',
                    description: 'Information retrieval mechanisms and RAG architectures',
                    category: 'systems',
                    tags: ['retrieval', 'rag', 'search']
                },
                {
                    id: 'fine-tuning-pipeline',
                    title: 'Fine-tuning Pipeline',
                    url: 'topic-fine-tuning-pipeline.html',
                    description: 'Model adaptation and domain-specific optimization',
                    category: 'training',
                    tags: ['fine-tuning', 'training', 'optimization']
                }
            ];

            defaultPages.forEach(pageData => {
                this.createPage(pageData);
            });

            console.log('Default pages created');
        }
    }

    // Content validation
    validatePage(page) {
        const errors = [];
        
        if (!page.title || page.title.trim().length === 0) {
            errors.push('Page title is required');
        }
        
        if (!page.url || page.url.trim().length === 0) {
            errors.push('Page URL is required');
        }
        
        if (page.url && !page.url.match(/^[a-zA-Z0-9-_]+\.html$/)) {
            errors.push('URL must be a valid HTML filename');
        }
        
        return {
            isValid: errors.length === 0,
            errors
        };
    }

    // SEO and metadata
    generateSEOMetadata(page) {
        return {
            title: page.title + ' - ' + this.settings.siteName,
            description: page.description,
            keywords: page.metadata.tags.join(', '),
            author: page.metadata.author,
            canonical: page.url,
            og: {
                title: page.title,
                description: page.description,
                type: 'article',
                url: page.url
            },
            twitter: {
                card: 'summary_large_image',
                title: page.title,
                description: page.description
            }
        };
    }

    // Export functionality
    exportPageHTML(pageId) {
        const page = this.getPage(pageId);
        if (!page) {
            throw new Error(`Page ${pageId} not found`);
        }

        const seoMetadata = this.generateSEOMetadata(page);
        
        const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${seoMetadata.title}</title>
    <meta name="description" content="${seoMetadata.description}">
    <meta name="keywords" content="${seoMetadata.keywords}">
    <meta name="author" content="${seoMetadata.author}">
    
    <!-- Open Graph -->
    <meta property="og:title" content="${seoMetadata.og.title}">
    <meta property="og:description" content="${seoMetadata.og.description}">
    <meta property="og:type" content="${seoMetadata.og.type}">
    <meta property="og:url" content="${seoMetadata.og.url}">
    
    <!-- Twitter Card -->
    <meta name="twitter:card" content="${seoMetadata.twitter.card}">
    <meta name="twitter:title" content="${seoMetadata.twitter.title}">
    <meta name="twitter:description" content="${seoMetadata.twitter.description}">
    
    <!-- Canonical -->
    <link rel="canonical" href="${seoMetadata.canonical}">
    
    <!-- Stylesheets -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Custom Styles -->
    <style>
        .content-box { margin: 1rem 0; padding: 1rem; border-radius: 0.5rem; }
        .box-note { background: #fef3c7; border-left: 4px solid #f59e0b; }
        .box-example { background: #dbeafe; border-left: 4px solid #3b82f6; }
        .box-warning { background: #fee2e2; border-left: 4px solid #ef4444; }
        .box-tip { background: #dcfce7; border-left: 4px solid #22c55e; }
        .box-info { background: #f3e8ff; border-left: 4px solid #8b5cf6; }
    </style>
</head>
<body>
    <div class="container mx-auto px-4 py-8">
        <h1>${page.title}</h1>
        <p>${page.description}</p>
        ${page.content}
    </div>
</body>
</html>`;

        return html;
    }

    // Performance monitoring
    getMetrics() {
        return {
            totalPages: this.pages.size,
            totalTemplates: this.templates.size,
            totalMedia: this.media.size,
            storageUsed: this.getStorageUsage(),
            lastBackup: localStorage.getItem('cms_last_backup'),
            lastActivity: localStorage.getItem('cms_last_activity')
        };
    }

    getStorageUsage() {
        let totalSize = 0;
        for (const key in localStorage) {
            if (key.startsWith('cms_')) {
                totalSize += localStorage[key].length;
            }
        }
        return this.formatFileSize(totalSize);
    }
}

// Initialize the CMS
const cms = new AIKnowledgeCMS();

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AIKnowledgeCMS;
}

// Global functions for the visual editor
window.CMS = cms;

// Utility functions for the visual editor
function initializeEditor() {
    console.log('Visual Editor initialized with CMS backend');
    
    // Load current page data
    const currentPageId = 'ai-stack-overview'; // Default page
    const page = cms.getPage(currentPageId);
    
    if (page) {
        window.currentPage = page;
        console.log('Loaded page:', page.title);
    }
}

function savePage() {
    try {
        cms.updatePage(window.currentPage.id, {
            title: document.getElementById('page-title').value,
            url: document.getElementById('page-url').value,
            description: document.getElementById('page-description').value,
            content: document.getElementById('editor-content').innerHTML,
            metadata: {
                ...window.currentPage.metadata,
                modified: new Date().toISOString()
            }
        });
        
        updateSaveStatus('Saved');
        showToast('Page saved successfully!', 'success');
        
        // Update local currentPage object
        window.currentPage = cms.getPage(window.currentPage.id);
    } catch (error) {
        console.error('Save failed:', error);
        showToast('Save failed: ' + error.message, 'error');
    }
}

function createNewPage() {
    const newPage = cms.createPage({
        title: 'New Page',
        url: 'new-page.html',
        description: 'New page description'
    });
    
    window.currentPage = newPage;
    loadCurrentPage();
    showToast('New page created!', 'success');
}

function exportPage() {
    try {
        const html = cms.exportPageHTML(window.currentPage.id);
        const blob = new Blob([html], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = window.currentPage.url;
        a.click();
        
        URL.revokeObjectURL(url);
        showToast('Page exported successfully!', 'success');
    } catch (error) {
        showToast('Export failed: ' + error.message, 'error');
    }
}

// Enhanced content management functions
function addContentBlock(pageId, type, content = '') {
    const block = cms.addContentBlock(pageId, {
        type: type,
        content: content,
        position: 0
    });
    
    return block;
}

function updateContentBlock(pageId, blockId, content) {
    cms.updateContentBlock(pageId, blockId, { content });
}

function deleteContentBlock(pageId, blockId) {
    cms.deleteContentBlock(pageId, blockId);
}

// Theme and design management
function setColorScheme(color) {
    cms.updateSetting('colorScheme', color);
    cms.applySettings();
    
    // Update UI elements
    document.querySelectorAll('.color-picker').forEach(picker => {
        picker.classList.remove('ring-4', 'ring-white');
    });
    
    event.target.classList.add('ring-4', 'ring-white');
}

function setFontFamily(font) {
    cms.updateSetting('fontFamily', font);
    cms.applySettings();
}

// Advanced features
function backupSystem() {
    const backup = cms.createBackup();
    showToast('System backup created!', 'success');
    return backup;
}

function getSystemMetrics() {
    return cms.getMetrics();
}

function searchContent(query) {
    return cms.searchPages(query);
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the editor with CMS
    if (typeof initializeEditor === 'function') {
        initializeEditor();
    }
    
    // Setup global CMS reference
    window.cms = cms;
    
    console.log('CMS System Ready');
    console.log('Total pages:', cms.getMetrics().totalPages);
});

console.log('Content Management System loaded successfully!');