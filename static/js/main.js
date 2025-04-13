document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const singleTab = document.getElementById('single-tab');
    const batchTab = document.getElementById('batch-tab');
    const singleContent = document.getElementById('single-content');
    const batchContent = document.getElementById('batch-content');

    singleTab.addEventListener('click', function() {
        singleTab.classList.add('active');
        batchTab.classList.remove('active');
        singleContent.classList.add('active');
        batchContent.classList.remove('active');
    });

    batchTab.addEventListener('click', function() {
        batchTab.classList.add('active');
        singleTab.classList.remove('active');
        batchContent.classList.add('active');
        singleContent.classList.remove('active');
    });

    // Single image upload functionality
    const fileInput = document.getElementById('file-input');
    const singlePreview = document.getElementById('single-preview');
    const singlePreviewContent = singlePreview.querySelector('.preview-content');
    const clearSingleBtn = document.getElementById('clear-single');
    const analyzeSingleBtn = document.getElementById('analyze-single');
    const singleUploadArea = document.querySelector('#single-upload .upload-area');

    // Batch image upload functionality
    const batchFileInput = document.getElementById('file-input-batch');
    const batchPreview = document.getElementById('batch-preview');
    const batchPreviewContent = batchPreview.querySelector('.preview-content');
    const clearBatchBtn = document.getElementById('clear-batch');
    const analyzeBatchBtn = document.getElementById('analyze-batch');
    const fileCountElement = document.getElementById('file-count');
    const batchUploadArea = document.querySelector('#batch-upload .upload-area');

    // Results containers
    const resultsContainer = document.getElementById('results-container');
    const resultsContent = document.getElementById('results-content');
    const closeResultsBtn = document.getElementById('close-results');
    const loader = document.getElementById('loader');

    // Batch results containers
    const batchResultsContainer = document.getElementById('batch-results-container');
    const batchSummary = document.getElementById('batch-summary');
    const batchResults = document.getElementById('batch-results');
    const closeBatchResultsBtn = document.getElementById('close-batch-results');
    const batchLoader = document.getElementById('batch-loader');
    const batchProgress = document.getElementById('batch-progress');
    const progressText = document.getElementById('progress-text');

    // Error modal
    const errorModal = document.getElementById('error-modal');
    const errorMessage = document.getElementById('error-message');
    const closeModal = document.querySelector('.close-modal');

    // Single file upload handlers
    fileInput.addEventListener('change', function(e) {
        handleFileSelect(e.target.files[0], singlePreviewContent, singlePreview);
        analyzeSingleBtn.disabled = false;
    });

    singleUploadArea.addEventListener('dragover', handleDragOver);
    singleUploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        handleFileSelect(e.dataTransfer.files[0], singlePreviewContent, singlePreview);
        analyzeSingleBtn.disabled = false;
    });

    clearSingleBtn.addEventListener('click', function() {
        singlePreviewContent.innerHTML = '';
        fileInput.value = '';
        singlePreview.style.display = 'none';
        analyzeSingleBtn.disabled = true;
    });

    // Batch file upload handlers
    batchFileInput.addEventListener('change', function(e) {
        handleBatchFileSelect(e.target.files, batchPreviewContent, batchPreview);
        analyzeBatchBtn.disabled = false;
    });

    batchUploadArea.addEventListener('dragover', handleDragOver);
    batchUploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        handleBatchFileSelect(e.dataTransfer.files, batchPreviewContent, batchPreview);
        analyzeBatchBtn.disabled = false;
    });

    clearBatchBtn.addEventListener('click', function() {
        batchPreviewContent.innerHTML = '';
        batchFileInput.value = '';
        batchPreview.style.display = 'none';
        fileCountElement.textContent = '0';
        analyzeBatchBtn.disabled = true;
    });

    // Analyze buttons handlers
    analyzeSingleBtn.addEventListener('click', analyzeImage);
    analyzeBatchBtn.addEventListener('click', analyzeBatch);

    // Close results buttons
    closeResultsBtn.addEventListener('click', function() {
        resultsContainer.style.display = 'none';
    });

    closeBatchResultsBtn.addEventListener('click', function() {
        batchResultsContainer.style.display = 'none';
    });

    // Close error modal
    closeModal.addEventListener('click', function() {
        errorModal.style.display = 'none';
    });

    window.addEventListener('click', function(e) {
        if (e.target === errorModal) {
            errorModal.style.display = 'none';
        }
    });

    // File handling functions
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('dragover');
    }

    function handleFileSelect(file, previewContent, previewContainer) {
        if (!file || !file.type.startsWith('image/')) {
            showError('Please select a valid image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            previewContent.innerHTML = `
                <div class="preview-item">
                    <img src="${e.target.result}" alt="Preview">
                    <p>${file.name}</p>
                </div>
            `;
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    function handleBatchFileSelect(files, previewContent, previewContainer) {
        if (!files || files.length === 0) {
            return;
        }

        previewContent.innerHTML = '';
        let validFiles = 0;

        for (let i = 0; i < files.length; i++) {
            if (files[i].type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const previewItem = document.createElement('div');
                    previewItem.className = 'preview-item';
                    previewItem.innerHTML = `
                        <img src="${e.target.result}" alt="Preview">
                        <p>${files[i].name}</p>
                    `;
                    previewContent.appendChild(previewItem);
                };
                reader.readAsDataURL(files[i]);
                validFiles++;
            }
        }

        if (validFiles > 0) {
            fileCountElement.textContent = validFiles;
            previewContainer.style.display = 'block';
        } else {
            showError('Please select valid image files');
        }
    }

    // API functions
    function analyzeImage() {
        if (!fileInput.files || fileInput.files.length === 0) {
            showError('Please select an image to analyze');
            return;
        }

        showLoader(resultsContainer, loader);
        
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoader(loader);
            if (data.success) {
                displayResults(data.results);
            } else {
                showError(data.error || 'Analysis failed');
            }
        })
        .catch(error => {
            hideLoader(loader);
            showError('Error: ' + error.message);
        });
    }

    function analyzeBatch() {
        if (!batchFileInput.files || batchFileInput.files.length === 0) {
            showError('Please select images to analyze');
            return;
        }

        showLoader(batchResultsContainer, batchLoader);
        
        const formData = new FormData();
        for (let i = 0; i < batchFileInput.files.length; i++) {
            formData.append('images[]', batchFileInput.files[i]);
        }

        fetch('/batch_analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoader(batchLoader);
            if (data.success) {
                displayBatchResults(data.results);
            } else {
                showError(data.error || 'Batch analysis failed');
            }
        })
        .catch(error => {
            hideLoader(batchLoader);
            showError('Error: ' + error.message);
        });
    }

    // UI helper functions
    function showLoader(container, loaderElement) {
        container.style.display = 'block';
        loaderElement.style.display = 'flex';
        document.body.classList.add('loading');
    }

    function hideLoader(loaderElement) {
        loaderElement.style.display = 'none';
        document.body.classList.remove('loading');
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorModal.style.display = 'block';
    }

    // This is the specific section of the displayResults function in main.js that needs to be changed.
// The rest of the file remains the same

function displayResults(results) {
    resultsContent.innerHTML = `
        <div class="result-box ${results.predicted_class}">
            <div class="result-image">
                <img src="${results.image_url}" alt="${results.original_filename}">
            </div>
            <div class="result-details">
                <div class="result-header ${results.predicted_class}">
                    <h3>
                        <i class="fas ${results.predicted_class === 'ai' ? 'fa-robot' : 'fa-user'}"></i>
                        ${results.predicted_class.toUpperCase()} Generated
                    </h3>
                    <div class="confidence-meter">
                        <div class="confidence-bar" style="width: ${Math.round(results.confidence * 100)}%"></div>
                        <span>${Math.round(results.confidence * 100)}% Confidence</span>
                    </div>
                </div>

                <div class="result-section">
                    <h4>Model Analysis</h4>
                    <div class="model-results">
                        ${Object.entries(results.model_results).map(([model, probs]) => `
                            <div class="model-result">
                                <div class="model-name">${model}</div>
                                <div class="model-probs">
                                    <div class="prob ai" style="width: ${Math.round(probs.ai * 100)}%"></div>
                                    <div class="prob human" style="width: ${Math.round(probs.human * 100)}%"></div>
                                </div>
                                <div class="prob-labels">
                                    <span>AI: ${Math.round(probs.ai * 100)}%</span>
                                    <span>Human: ${Math.round(probs.human * 100)}%</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                ${results.metadata_analysis ? `
                    <div class="result-section">
                        <h4>Metadata Analysis</h4>
                        <div class="metadata-analysis">
                            <div class="metadata-signs">
                                ${results.metadata_analysis.ai_signs.length > 0 ? `
                                    <div class="sign-list ai-signs">
                                        <h5>AI Indicators</h5>
                                        <ul>
                                            ${results.metadata_analysis.ai_signs.map(sign => `<li>${sign}</li>`).join('')}
                                        </ul>
                                    </div>
                                ` : ''}
                                ${results.metadata_analysis.human_signs.length > 0 ? `
                                    <div class="sign-list human-signs">
                                        <h5>Human Indicators</h5>
                                        <ul>
                                            ${results.metadata_analysis.human_signs.map(sign => `<li>${sign}</li>`).join('')}
                                        </ul>
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                ` : ''}

                ${results.explanation ? `
                    <div class="result-section">
                        <h4>Explanation</h4>
                        <p class="explanation-summary">${results.explanation.summary}</p>
                        <div class="inference-time">
                            <h5>Processing Time</h5>
     <p class="inference-value">
        ${results.inference_time ? (results.inference_time.toFixed(3) + ' seconds') : 'Not available'}
    </p>
                        </div>
                    </div>
                ` : ''}

                <div class="result-actions">
                    <a href="${results.report_url}" class="action-btn" target="_blank">
                        <i class="fas fa-file-alt"></i> View Detailed Report
                    </a>
                </div>
            </div>
        </div>
    `;

    resultsContainer.style.display = 'block';
}
    function displayBatchResults(results) {
        // Display batch summary
        batchSummary.innerHTML = `
            <div class="summary-stats">
                <div class="stat-box">
                    <div class="stat-value">${results.total}</div>
                    <div class="stat-label">Total Images</div>
                </div>
                <div class="stat-box ai">
                    <div class="stat-value">${results.ai}</div>
                    <div class="stat-label">AI Generated</div>
                </div>
                <div class="stat-box human">
                    <div class="stat-value">${results.human}</div>
                    <div class="stat-label">Human Photos</div>
                </div>
                <div class="stat-box failed">
                    <div class="stat-value">${results.failed}</div>
                    <div class="stat-label">Failed Analysis</div>
                </div>
            </div>
            <div class="summary-actions">
                <a href="${results.report_url}" class="action-btn" target="_blank">
                    <i class="fas fa-file-download"></i> Download Full Report
                </a>
            </div>
        `;

        // Display individual results
        batchResults.innerHTML = `
            <div class="batch-results-grid">
                ${results.results.map(result => `
                    <div class="batch-result-item ${result.predicted_class}">
                        <div class="result-image">
                            <img src="${result.image_url}" alt="${result.filename}">
                        </div>
                        <div class="result-info">
                            <div class="result-filename">${result.filename}</div>
                            <div class="result-prediction">
                                <span class="prediction-label ${result.predicted_class}">
                                    ${result.predicted_class.toUpperCase()}
                                </span>
                                <span class="prediction-confidence">
                                    ${Math.round(result.confidence * 100)}%
                                </span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        batchResultsContainer.style.display = 'block';
    }
});