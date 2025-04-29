document.querySelector('form').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(this);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Received data:', data);
        const dataContainer = document.getElementById('data-container');
        if (data && data.data_points) {
            data.data_points.forEach(item => {
                const p = document.createElement('p');
                p.textContent = item;
                dataContainer.appendChild(p);
            });
        } else if (data && data.result) {
            const p = document.createElement('p');
            p.textContent = data.result;
            dataContainer.appendChild(p);
        } else if (data && data.error) {
            const p = document.createElement('p');
            p.textContent = 'Error: ' + data.error;
            dataContainer.appendChild(p);
        }
    })
    .catch(error => {
        console.error('Error uploading file:', error);
        const dataContainer = document.getElementById('data-container');
        const p = document.createElement('p');
        p.textContent = 'Error uploading file.';
        dataContainer.appendChild(p);
    });
});