            viewTopic(topicId) {
                const topic = this.topics.find(t => t.id === topicId);
                if (!topic) return;

                // Navigate to the actual HTML page instead of showing modal
                if (topic.url) {
                    window.location.href = topic.url;
                } else {
                    // Fallback to modal for topics without direct URLs
                    this.currentTopic = topic;
                    this.populateModal(topic);
                    this.showModal();
                }
            }