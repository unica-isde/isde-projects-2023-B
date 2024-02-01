$(document).ready(function () {
    var scripts = document.getElementById('makeHistogram');
    var histogram = scripts.getAttribute('histogram');
    // L'immagine è stata caricata, ora puoi eseguire la funzione makeHistogram
    makeHistogram(histogram);
});

function makeHistogram(counts) {
    console.log(counts);
    counts = JSON.parse(counts);
    // Crea un oggetto dati per il grafico
    var data = {
    labels: Array.from({ length: counts.length }, (_, i) => i),  // Etichette per i bin
    datasets: [{
        label: 'Istogramma',
        data: counts,  // I dati dell'istogramma
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1
    }]
    };
    // Crea un oggetto di opzioni per il grafico (puoi personalizzare ulteriormente le opzioni)
    var options = {
    scales: {
        y: {
            beginAtZero: true
        }
    }
    };
    // Ottieni il riferimento all'elemento canvas dove verrà visualizzato il grafico
    var ctx = document.getElementById("histogramOutput").getContext('2d');
    // Crea il grafico a istogramma utilizzando Chart.js
    var istogramma = new Chart(ctx, {
        type: 'bar',
        data: data,
        options: options
    });
}