// popup.js

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = 'AIzaSyBctFb2dhVeDAW6GMdcmPmGKyoSpiR6lag';
  // const API_URL = 'http://my-elb-2062136355.us-east-1.elb.amazonaws.com:80';   
  const API_URL = 'http://localhost:5000';  // Remove trailing slash
  //const API_URL = 'http://23.20.221.231:8080';

  // Show loading state
  function showLoading(message) {
    return `
      <div class="loading">
        <div class="loading-spinner"></div>
        <span>${message}</span>
      </div>
    `;
  }

  // Show error state
  function showError(message) {
    return `<div class="error"> ${message}</div>`;
  }

  // Show success state
  function showSuccess(message) {
    return `<div class="success"> ${message}</div>`;
  }

  // Get sentiment class for styling
  function getSentimentClass(sentiment) {
    switch(sentiment) {
      case "1": return "positive";
      case "-1": return "negative";
      case "0": return "neutral";
      default: return "neutral";
    }
  }

  // Get sentiment label
  function getSentimentLabel(sentiment) {
    switch(sentiment) {
      case "1": return "Positive";
      case "-1": return "Negative";
      case "0": return "Neutral";
      default: return "Neutral";
    }
  }

  // Get the current tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      
      // Show initial loading state
      outputDiv.innerHTML = `
        <div class="section-title">YouTube Video Analysis</div>
        <p><strong>Video ID:</strong> ${videoId}</p>
        ${showLoading('Fetching comments...')}
      `;

      const comments = await fetchComments(videoId);
      if (comments.length === 0) {
        outputDiv.innerHTML = `
          <div class="section-title">YouTube Video Analysis</div>
          <p><strong>Video ID:</strong> ${videoId}</p>
          ${showError('No comments found for this video.')}
        `;
        return;
      }

      // Update loading state
      outputDiv.innerHTML = `
        <div class="section-title">YouTube Video Analysis</div>
        <p><strong>Video ID:</strong> ${videoId}</p>
        ${showSuccess(`Found ${comments.length} comments!`)}
        ${showLoading('Analyzing sentiment...')}
      `;

      const predictions = await getSentimentPredictions(comments);

      if (predictions) {
        // Process the predictions to get sentiment counts and sentiment data
        const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
        const sentimentData = []; // For trend graph
        const totalSentimentScore = predictions.reduce((sum, item) => sum + parseInt(item.sentiment), 0);
        predictions.forEach((item, index) => {
          sentimentCounts[item.sentiment]++;
          sentimentData.push({
            timestamp: item.timestamp,
            sentiment: parseInt(item.sentiment)
          });
        });

        // Compute metrics
        const totalComments = comments.length;
        const uniqueCommenters = new Set(comments.map(comment => comment.authorId)).size;
        const totalWords = comments.reduce((sum, comment) => sum + comment.text.split(/\s+/).filter(word => word.length > 0).length, 0);
        const avgWordLength = (totalWords / totalComments).toFixed(2);
        const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);

        // Normalize the average sentiment score to a scale of 0 to 10
        const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

        // Clear and rebuild the output
        outputDiv.innerHTML = `
          <div class="section-title">YouTube Video Analysis</div>
          <p><strong>Video ID:</strong> ${videoId}</p>
          ${showSuccess(`Analysis complete! Found ${comments.length} comments.`)}
        `;

        // Add the Comment Analysis Summary section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Comment Analysis Summary</div>
            <div class="metrics-container">
              <div class="metric">
                <div class="metric-title">Total Comments</div>
                <div class="metric-value">${totalComments}</div>
              </div>
              <div class="metric">
                <div class="metric-title">Unique Commenters</div>
                <div class="metric-value">${uniqueCommenters}</div>
              </div>
              <div class="metric">
                <div class="metric-title">Avg Comment Length</div>
                <div class="metric-value">${avgWordLength}</div>
              </div>
              <div class="metric">
                <div class="metric-title">Sentiment Score</div>
                <div class="metric-value">${normalizedSentimentScore}/10</div>
              </div>
            </div>
          </div>
        `;

        // Add the Sentiment Analysis Results section with a placeholder for the chart
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Sentiment Distribution</div>
            <p>Interactive pie chart showing comment sentiment breakdown</p>
            <div id="chart-container">${showLoading('Generating chart...')}</div>
          </div>`;

        // Fetch and display the pie chart inside the chart-container div
        await fetchAndDisplayChart(sentimentCounts);

        // Add the Sentiment Trend Graph section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Sentiment Trend Over Time</div>
            <p>Track how sentiment changes over time</p>
            <div id="trend-graph-container">${showLoading('Generating trend graph...')}</div>
          </div>`;

        // Fetch and display the sentiment trend graph
        await fetchAndDisplayTrendGraph(sentimentData);

        // Add the Word Cloud section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Comment Word Cloud</div>
            <p>Most frequently used words in comments</p>
            <div id="wordcloud-container">${showLoading('Generating word cloud...')}</div>
          </div>`;

        // Fetch and display the word cloud inside the wordcloud-container div
        await fetchAndDisplayWordCloud(comments.map(comment => comment.text));

        // Add the top comments section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Top Comments with Sentiment</div>
            <p>Most recent comments with AI-powered sentiment analysis</p>
            <ul class="comment-list">
              ${predictions.slice(0, 25).map((item, index) => `
                <li class="comment-item">
                  <div class="comment-text">${index + 1}. ${item.comment}</div>
                  <span class="comment-sentiment ${getSentimentClass(item.sentiment)}">${getSentimentLabel(item.sentiment)}</span>
                </li>`).join('')}
            </ul>
          </div>`;
      }
    } else {
      outputDiv.innerHTML = `
        <div class="section-title">Invalid URL</div>
        ${showError('Please navigate to a YouTube video to analyze comments.')}
      `;
    }
  });

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 500) {
        const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
        const data = await response.json();
        if (data.items) {
          data.items.forEach(item => {
            const commentText = item.snippet.topLevelComment.snippet.textOriginal;
            const timestamp = item.snippet.topLevelComment.snippet.publishedAt;
            const authorId = item.snippet.topLevelComment.snippet.authorChannelId?.value || 'Unknown';
            comments.push({ text: commentText, timestamp: timestamp, authorId: authorId });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      const outputDiv = document.getElementById("output");
      outputDiv.innerHTML += showError("Error fetching comments from YouTube API.");
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      // First check if the API is healthy
      try {
        const healthResponse = await fetch(`${API_URL}/health`);
        if (!healthResponse.ok) {
          throw new Error(`Health check failed: ${healthResponse.status}`);
        }
        const healthData = await healthResponse.json();
        console.log('API Health:', healthData);
        
        if (!healthData.model_loaded) {
          throw new Error('Model not loaded on server');
        }
      } catch (healthError) {
        console.error("Health check failed:", healthError);
        const outputDiv = document.getElementById("output");
        outputDiv.innerHTML += showError(`API health check failed: ${healthError.message}`);
      }
      
      const response = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server error response:', errorText);
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }
      
      const result = await response.json();
      console.log('Prediction result:', result);
      return result;
    } catch (error) {
      console.error("Error fetching predictions:", error);
      const outputDiv = document.getElementById("output");
      outputDiv.innerHTML += showError(`Error fetching sentiment predictions: ${error.message}`);
      
      // Add debugging information
      outputDiv.innerHTML += `
        <div class="section" style="margin-top: 10px;">
          <div class="section-title">🔧 Debug Information</div>
          <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 8px; font-size: 12px; color: rgba(255, 255, 255, 0.8);">
            <p><strong>API URL:</strong> ${API_URL}</p>
            <p><strong>Error:</strong> ${error.message}</p>
            <p><strong>Comments count:</strong> ${comments.length}</p>
            <p>Please check if the backend server is running and accessible.</p>
          </div>
        </div>
      `;
      
      return null;
    }
  }

  async function fetchAndDisplayChart(sentimentCounts) {
    try {
      const response = await fetch(`${API_URL}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts })
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.style.width = '100%';
      img.style.marginTop = '20px';
      // Append the image to the chart-container div
      const chartContainer = document.getElementById('chart-container');
      chartContainer.innerHTML = '';
      chartContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching chart image:", error);
      const chartContainer = document.getElementById('chart-container');
      chartContainer.innerHTML = showError(`Error loading chart: ${error.message}`);
    }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.style.width = '100%';
      img.style.marginTop = '20px';
      // Append the image to the wordcloud-container div
      const wordcloudContainer = document.getElementById('wordcloud-container');
      wordcloudContainer.innerHTML = '';
      wordcloudContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching word cloud image:", error);
      const wordcloudContainer = document.getElementById('wordcloud-container');
      wordcloudContainer.innerHTML = showError(`Error loading word cloud: ${error.message}`);
    }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const response = await fetch(`${API_URL}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData })
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.style.width = '100%';
      img.style.marginTop = '20px';
      // Append the image to the trend-graph-container div
      const trendGraphContainer = document.getElementById('trend-graph-container');
      trendGraphContainer.innerHTML = '';
      trendGraphContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching trend graph image:", error);
      const trendGraphContainer = document.getElementById('trend-graph-container');
      trendGraphContainer.innerHTML = showError(`Error loading trend graph: ${error.message}`);
    }
  }
});