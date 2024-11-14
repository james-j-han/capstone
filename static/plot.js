fetch("/plot-data")
  .then((response) => response.json())
  .then((data) => {
    const dots = data.map((d) => ({
      x: d.x,
      y: d.y,
      z: d.z, // Add z-coordinate for the 3D plot
      image: d.image,
    }));

    // 2D Plot with images
    const scatter2d = {
      x: dots.map((d) => d.x),
      y: dots.map((d) => d.y),
      mode: "markers",
      marker: { size: 5, color: "green" },
      type: "scatter",
    };

    const layout2d = {
      xaxis: { title: "X Axis" },
      yaxis: { title: "Y Axis" },
      autosize: true,
      responsive: true,
      margin: { t: 50, l: 50, r: 50, b: 50 },
    };

    Plotly.newPlot("plot-2d", [scatter2d], layout2d);

    // Handle zoom events for 2D plot
    document.getElementById("plot-2d").on("plotly_relayout", (eventData) => {
      if (eventData["xaxis.range[0]"] !== undefined) {
        // Get the current visible range
        const xRange = Math.abs(
          eventData["xaxis.range[1]"] - eventData["xaxis.range[0]"]
        );
        const yRange = Math.abs(
          eventData["yaxis.range[1]"] - eventData["yaxis.range[0]"]
        );

        // Calculate the dynamic zoom threshold
        const totalXRange =
          Math.max(...dots.map((d) => d.x)) - Math.min(...dots.map((d) => d.x));
        const totalYRange =
          Math.max(...dots.map((d) => d.y)) - Math.min(...dots.map((d) => d.y));
        const zoomThreshold = Math.min(totalXRange, totalYRange) * 0.1; // 10% of the dataset range

        if (xRange < zoomThreshold && yRange < zoomThreshold) {
          // Display images
          const imageAnnotations = dots.map((d) => ({
            x: d.x,
            y: d.y,
            xref: "x",
            yref: "y",
            xanchor: "center",
            yanchor: "middle",
            sizex: 0.1,
            sizey: 0.1,
            source: d.image,
            layer: "below",
          }));

          const layoutUpdate = { images: imageAnnotations };
          Plotly.relayout("plot-2d", layoutUpdate);

          // Hide dots
          Plotly.restyle("plot-2d", { visible: false }, [0]); // Trace 0 is the scatter
        } else {
          // Reset to dots
          Plotly.relayout("plot-2d", { images: [] }); // Clear images
          Plotly.restyle("plot-2d", { visible: true }, [0]);
        }
      }
    });

    // 3D Plot with gradient color
    const scatter3d = {
      x: dots.map((d) => d.x),
      y: dots.map((d) => d.y),
      z: dots.map((d) => d.z), // Z-coordinates
      mode: "markers",
      marker: {
        size: 5,
        color: dots.map((d) => d.z), // Gradient based on z-values
        colorscale: "Viridis", // Gradient color scheme
        opacity: 0.8,
      },
      type: "scatter3d",
    };

    const layout3d = {
      scene: {
        xaxis: { title: "X Axis" },
        yaxis: { title: "Y Axis" },
        zaxis: { title: "Z Axis" },
      },
      autosize: true,
      responsive: true,
      margin: { t: 50, l: 50, r: 50, b: 50 },
    };

    Plotly.newPlot("plot-3d", [scatter3d], layout3d);
  })
  .catch((error) => console.error("Error fetching plot data:", error));

document.querySelectorAll('input[name="query-type"]').forEach((radio) => {
  radio.addEventListener("change", function () {
    if (this.value === "image") {
      document.getElementById("image-query-form").style.display = "block";
      document.getElementById("text-query-form").style.display = "none";
    } else {
      document.getElementById("image-query-form").style.display = "none";
      document.getElementById("text-query-form").style.display = "block";
    }
  });
});

// Image Query Submission
document
  .getElementById("image-query-form")
  .addEventListener("submit", function (event) {
    event.preventDefault();

    const formData = new FormData(this);
    formData.append("type", "image");

    fetch("/query", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => handleQueryResults(data))
      .catch((error) => console.error("Error querying image:", error));
  });

// Text Query Submission
document
  .getElementById("text-query-form")
  .addEventListener("submit", function (event) {
    event.preventDefault();

    // const formData = new FormData(this);
    const queryText = this.text.value;
    const k = this.k.value;

    const payload = { type: "text", text: queryText, k };

    console.log("Text query payload:", payload);

    fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Response from server:", data);
        handleQueryResults(data, k);
      })
      .catch((error) => console.error("Error querying text:", error));
  });

// Handle Query Results
function handleQueryResults(data, kValue) {
  if (data.error) {
    console.error("Server Error:", data.error);
    return;
  }

  // Clear previous results
  const resultsDiv = document.getElementById("query-results");
  resultsDiv.innerHTML = "";

  const title = document.createElement("h5");
  title.className = "text-primary mb-3"; // Styling for the title
  title.innerText = `Top-${kValue} Results`;
  resultsDiv.appendChild(title);

  // Display top-K images
  if (data.top_k_results) {
    data.top_k_results.forEach((item) => {
      const imgElement = document.createElement("img");
      imgElement.src = item.image;
      imgElement.style.width = "100px";
      imgElement.style.margin = "10px";
      resultsDiv.appendChild(imgElement);
    });
  }

  // Plot query point on 2D plot
  if (
    data.query_point &&
    data.query_point.x_2d !== null &&
    data.query_point.y_2d !== null
  ) {
    const queryName =
      data.query_type == "text" ? `${data.query_text}` : "Uploaded Image";
    console.log("Query Type:", data.query_type);
    console.log("Query Text:", data.query_text);

    Plotly.addTraces("plot-2d", {
      x: [data.query_point.x_2d],
      y: [data.query_point.y_2d],
      mode: "markers",
      marker: {
        size: 5,
        color: "red", // Distinguish query point
      },
      name: queryName,
    });
  }

  // Plot query point on 3D plot
  if (
    data.query_point &&
    data.query_point.x_3d !== null &&
    data.query_point.y_3d !== null &&
    data.query_point.z_3d !== null
  ) {
    Plotly.addTraces("plot-3d", {
      x: [data.query_point.x_3d],
      y: [data.query_point.y_3d],
      z: [data.query_point.z_3d],
      mode: "markers",
      marker: {
        size: 5,
        color: "red",
      },
      name: "Query Image/Text",
    });
  }
}

// Function to dynamically populate a select element
function populateSelect(selectId, maxOptions) {
  const selectElement = document.getElementById(selectId);

  for (let i = 1; i <= maxOptions; i++) {
    const option = document.createElement("option");
    option.value = i;
    option.textContent = i; // Set the text inside the option
    if (i === 5) option.selected = true; // Default selected value
    selectElement.appendChild(option);
  }
}

// Populate both Top-K dropdowns
populateSelect("k-image", 20); // For the image query form
populateSelect("k-text", 20); // For the text query form

function toggleForm(queryType) {
  const imageForm = document.getElementById("image-query-form");
  const textForm = document.getElementById("text-query-form");

  if (queryType === "image") {
    imageForm.style.display = "block";
    textForm.style.display = "none";
  } else if (queryType === "text") {
    imageForm.style.display = "none";
    textForm.style.display = "block";
  } else {
    console.error("Invalid query type:", queryType);
  }
}

// document.querySelectorAll('select[name="k"]').forEach((select) => {
//   select.addEventListener("change", function () {
//     console.log(`K Value Changed: ${this.value}`);
//   });
// });
