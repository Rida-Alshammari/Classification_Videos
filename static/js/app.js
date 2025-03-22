const dragArea = document.getElementById('dragArea');
const fileInput = document.getElementById('fileInput');
const progressBar = document.getElementById('progressBar');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileDetails = document.getElementById('fileDetails');

dragArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dragArea.classList.add('highlight');
});

dragArea.addEventListener('dragleave', () => {
    dragArea.classList.remove('highlight');
});

dragArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dragArea.classList.remove('highlight');
    fileInput.files = e.dataTransfer.files;
    handleFile();
});

dragArea.addEventListener('click', () => fileInput.click());
document.getElementById('uploadBtn').addEventListener('click', function () {
    const file = fileInput.files[0];
    if (file) {
        uploadFile(file);
    } else {
        alert('Please select a video file first.');
    }
});
fileInput.addEventListener('change', handleFile);

function handleFile() {
    const file = fileInput.files[0];
    if (file && file.type.startsWith('video/')) {
        fileName.textContent = file.name;
        const size = (file.size / (1024 * 1024)).toFixed(2);
        fileDetails.textContent = `${size}MB • ${file.type.split('/')[1].toUpperCase()}`;
        fileInfo.style.display = 'flex';
    }
}
function uploadFile(file) {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);

    // Todo: Setup progress bar
    progressBar.style.width = '0%';
    progressBar.style.backgroundColor = '#7c00fe';
    progressBar.style.transition = 'width 0.3s ease';

    xhr.upload.addEventListener('progress', function (e) {
        if (e.lengthComputable) {
            const percent = (e.loaded / e.total) * 100;
            progressBar.style.width = percent + '%';

            const progressText = document.querySelector('.progress-text');
            if (progressText) {
                progressText.textContent = Math.round(percent) + '%';
            }
            // Todo: Change progress bar color
            if (percent >= 100) {
                progressBar.style.backgroundColor = '#17a2b8';
            }
        }
    });

    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                // Todo: Replace the entire HTML content
                document.documentElement.innerHTML = xhr.responseText;
            } else {
                progressBar.style.backgroundColor = '#dc3545';
                alert('Error: ' + xhr.statusText);
            }
        }
    };
    xhr.open('POST', '/upload');
    xhr.send(formData);
}