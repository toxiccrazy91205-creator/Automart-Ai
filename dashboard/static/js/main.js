document.addEventListener('DOMContentLoaded', function() {
    // Tab functionality
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const target = this.dataset.target;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            this.classList.add('active');
            document.getElementById(target).classList.add('active');
        });
    });

    // LSTM prediction button
    const lstmBtn = document.getElementById('lstm-btn');
    if (lstmBtn) {
        lstmBtn.addEventListener('click', function() {
            const resultDiv = document.getElementById('lstm-result');
            const btn = this;
            btn.disabled = true;
            btn.textContent = 'Training LSTM model...';

            fetch('/run_lstm/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCookie('csrftoken'),
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                btn.disabled = false;
                btn.textContent = '▶ Run LSTM Prediction';
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div class="alert alert-success">✅ Prediction complete in ${data.time}s</div>
                        <div class="metric-card">
                            <div class="label">Predicted Next-Day Sales</div>
                            <div class="value">${data.prediction} units</div>
                        </div>
                    `;
                    location.reload();
                } else {
                    resultDiv.innerHTML = `<div class="alert alert-error">❌ LSTM failed: ${data.error}</div>`;
                }
            })
            .catch(err => {
                btn.disabled = false;
                btn.textContent = '▶ Run LSTM Prediction';
                resultDiv.innerHTML = `<div class="alert alert-error">❌ Error: ${err}</div>`;
            });
        });
    }
});

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
