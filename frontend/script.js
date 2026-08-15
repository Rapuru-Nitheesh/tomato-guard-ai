// ============================================================
// TOMATO GUARD AI
// Frontend JavaScript
// ============================================================


// ============================================================
// ELEMENTS
// ============================================================

const imageInput =
    document.getElementById("imageInput");

const dropArea =
    document.getElementById("dropArea");

const previewContainer =
    document.getElementById(
        "previewContainer"
    );

const previewImage =
    document.getElementById(
        "previewImage"
    );

const removeButton =
    document.getElementById(
        "removeButton"
    );

const analyzeButton =
    document.getElementById(
        "analyzeButton"
    );

const loading =
    document.getElementById(
        "loading"
    );

const resultSection =
    document.getElementById(
        "resultSection"
    );

const prediction =
    document.getElementById(
        "prediction"
    );

const confidenceText =
    document.getElementById(
        "confidenceText"
    );

const confidenceBar =
    document.getElementById(
        "confidenceBar"
    );

const statusMessage =
    document.getElementById(
        "statusMessage"
    );

const topPredictions =
    document.getElementById(
        "topPredictions"
    );


// ============================================================
// BACKEND URL
// ============================================================

const API_URL =
    "https://tomato-guard-ai-eqzo.onrender.com";

// ============================================================
// SELECTED FILE
// ============================================================

let selectedFile = null;


// ============================================================
// FILE INPUT
// ============================================================

imageInput.addEventListener(
    "change",
    function () {

        const file =
            imageInput.files[0];

        if (file) {

            handleFile(file);

        }

    }
);


// ============================================================
// DRAG OVER
// ============================================================

dropArea.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        dropArea.classList.add(
            "dragover"
        );

    }
);


// ============================================================
// DRAG LEAVE
// ============================================================

dropArea.addEventListener(
    "dragleave",
    function () {

        dropArea.classList.remove(
            "dragover"
        );

    }
);


// ============================================================
// DROP
// ============================================================

dropArea.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        dropArea.classList.remove(
            "dragover"
        );


        const file =
            event.dataTransfer.files[0];


        if (file) {

            handleFile(file);

        }

    }
);


// ============================================================
// HANDLE FILE
// ============================================================

function handleFile(file) {

    // --------------------------------------------------------
    // Validate image
    // --------------------------------------------------------

    if (
        !file.type.startsWith(
            "image/"
        )
    ) {

        alert(
            "Please upload a valid image."
        );

        return;

    }


    // --------------------------------------------------------
    // Save selected file
    // --------------------------------------------------------

    selectedFile = file;


    // --------------------------------------------------------
    // Preview
    // --------------------------------------------------------

    const reader =
        new FileReader();


    reader.onload =
        function (event) {

            previewImage.src =
                event.target.result;


            previewContainer
                .classList
                .remove("hidden");


            analyzeButton
                .classList
                .remove("hidden");


            resultSection
                .classList
                .add("hidden");

        };


    reader.readAsDataURL(file);

}


// ============================================================
// REMOVE IMAGE
// ============================================================

removeButton.addEventListener(
    "click",
    function () {

        selectedFile = null;

        imageInput.value = "";

        previewImage.src = "";

        previewContainer
            .classList
            .add("hidden");

        analyzeButton
            .classList
            .add("hidden");

        resultSection
            .classList
            .add("hidden");

    }
);


// ============================================================
// ANALYZE BUTTON
// ============================================================

analyzeButton.addEventListener(
    "click",
    analyzeImage
);


// ============================================================
// ANALYZE IMAGE
// ============================================================

async function analyzeImage() {

    // --------------------------------------------------------
    // Check image
    // --------------------------------------------------------

    if (!selectedFile) {

        alert(
            "Please select a tomato leaf image first."
        );

        return;

    }


    // --------------------------------------------------------
    // UI: Loading
    // --------------------------------------------------------

    analyzeButton.disabled = true;

    loading
        .classList
        .remove("hidden");

    resultSection
        .classList
        .add("hidden");


    // --------------------------------------------------------
    // FormData
    // --------------------------------------------------------

    const formData =
        new FormData();


    formData.append(
        "image",
        selectedFile
    );


    try {

        // ----------------------------------------------------
        // Send image to Flask
        // ----------------------------------------------------

        const response =
            await fetch(
                `${API_URL}/predict`,
                {
                    method: "POST",
                    body: formData
                }
            );


        // ----------------------------------------------------
        // Read response
        // ----------------------------------------------------

        const data =
            await response.json();


        // ----------------------------------------------------
        // Check response
        // ----------------------------------------------------

        if (!response.ok ||
            !data.success) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );

        }


        // ----------------------------------------------------
        // Display result
        // ----------------------------------------------------

        displayResult(data);


    }
    catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        alert(
            "Unable to analyze the image.\n\n" +
            "Please make sure the Flask backend " +
            "is running."
        );

    }
    finally {

        loading
            .classList
            .add("hidden");

        analyzeButton.disabled = false;

    }

}


// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(data) {

    // --------------------------------------------------------
    // Prediction
    // --------------------------------------------------------

    prediction.textContent =
        `🍅 ${data.prediction}`;


    // --------------------------------------------------------
    // Confidence
    // --------------------------------------------------------

    const confidence =
        Number(
            data.confidence
        );


    confidenceText.textContent =
        `${confidence.toFixed(2)}%`;


    confidenceBar.style.width =
        `${confidence}%`;


    // --------------------------------------------------------
    // Status
    // --------------------------------------------------------

    if (data.healthy) {

        statusMessage.innerHTML = `

            🌿
            <strong>
                Healthy Tomato Leaf
            </strong>

            <br>

            The model classified this
            image as a healthy tomato leaf.

        `;

    }

    else if (data.low_confidence) {

        statusMessage.innerHTML = `

            ⚠️
            <strong>
                Low Confidence
            </strong>

            <br>

            The model is not highly confident
            about this prediction. Try uploading
            a clearer tomato leaf image.

        `;

    }

    else {

        statusMessage.innerHTML = `

            🔎
            <strong>
                Possible Condition Detected
            </strong>

            <br>

            This is an AI-based preliminary
            indication. Inspect the plant carefully
            and seek expert confirmation when needed.

        `;

    }


    // --------------------------------------------------------
    // Top predictions
    // --------------------------------------------------------

    topPredictions.innerHTML = "";


    data.top_predictions.forEach(
        function (item, index) {

            const row =
                document.createElement(
                    "div"
                );


            row.className =
                "prediction-row";


            const medal =
                index === 0
                    ? "🥇"
                    : index === 1
                        ? "🥈"
                        : "🥉";


            row.innerHTML = `

                <span>
                    ${medal}
                    ${item.name}
                </span>

                <strong>
                    ${Number(
                        item.confidence
                    ).toFixed(2)}%
                </strong>

            `;


            topPredictions.appendChild(
                row
            );

        }
    );


    // --------------------------------------------------------
    // Show result
    // --------------------------------------------------------

    resultSection
        .classList
        .remove("hidden");


    // --------------------------------------------------------
    // Scroll to result
    // --------------------------------------------------------

    resultSection.scrollIntoView({

        behavior: "smooth",

        block: "start"

    });

}
