


$(document).ready(function() {
    // Function to fetch and render initial list of stocks
    function fetchInitialStocks() {
        $.ajax({
            url: '/ticker/',
            dataType: 'json',
            success: function(data) {
                renderStocksTable(data.results);
            }
        });
    }

    // Function to render stocks in table format
    function renderStocksTable(stocks) {
        var resultsHtml = '<table class="table text-start align-middle table-bordered table-hover mb-0">' +
                            '<thead>' +
                                '<tr class="text-white">' +
                                    '<th scope="col">Ticker Symbol</th>' +
                                    '<th scope="col">Ticker Name</th>' +
                                    '<th scope="col">predict for 30 days</th>' +
                                '</tr>' +
                            '</thead>' +
                            '<tbody>';

        stocks.forEach(function(stock) {
            resultsHtml += '<tr>' +
                              '<td>' + stock.symbol + '</td>' +
                              '<td>' + stock.name + '</td>' +
                              '<td>' + `<a href="/predict/${stock.symbol.toLowerCase()}/30/" class ="btn btn-success">predict</a>` + '</td>' +

                            //   '<td>' + stock.name + '</td>' +
                           '</tr>';
        });

        resultsHtml += '</tbody></table>';

        $('#search-results').html(resultsHtml);
    }

    // Fetch and render initial stocks when the page loads
    fetchInitialStocks();

    // Search functionality
    $('#search-input').on('input', function() {
        var query = $(this).val();
        $.ajax({
            url: '/ticker/',
            data: {'query': query},
            dataType: 'json',
            success: function(data) {
                if (data ==''){
                    fetchInitialStocks();
                }
                else{
                    renderStocksTable(data.results);
                }
            }
        });
    });
});
// Assuming you have an input field with id="search-input" and a div with id="ticker-list"

console.log("Script loaded");
let searchInput = document.getElementById('search-input')
let tickerList = document.getElementById('ticker-list')

searchInput.addEventListener('input', function() {
    console.log("Input detected");
    if (searchInput.value.trim() !== '') {
        tickerList.style.display = 'none'; // Hide the initial list when user starts typing
    } else {
        tickerList.style.display = 'block'; // Show the initial list if input is empty
    }
});

// windowreload code when revisiting the page
window.addEventListener('pageshow', function(event) {
    var historyTraversal = event.persisted || (typeof window.performance != 'undefined' && window.performance.navigation.type === 2);
    if (historyTraversal) {
        // Reload the page
        window.location.reload();
    }
});
