
// class labels in order (to map index -> label)
const model_labels = ['rock', 'paper', 'scissors'];

const target_shape = [1, 200, 200, 3];

window.model = null;

function classify(img_element, should_alert) {

    let image = tf.browser
        .fromPixels(img_element)
        .reshape(target_shape);

    
    window.model
        .predict(image, {'batch_size': 1})
        .data()
        .then(data => {
            let idx = tf.argMax(data).dataSync();
            let predicted_label = model_labels[idx];
            document.getElementById("msg").hidden = true;

            console.log(`Predicted: ${idx}: ${predicted_label}`);
            if (should_alert) {
                alert(`Model prediction: ${predicted_label}`);

            }
        })
}


async function main() {
    
    if (!window.model) {
        window.model = await tf.loadLayersModel('http://0.0.0.0:8080/tfjs/model.json');
    }


    // ['rock', 'paper', 'scissors'].forEach(groundtruth => {

    //     console.log("Predicting... " + groundtruth);

    //     let img_element = document.getElementById(groundtruth);
    //     classify(img_element, false);

    // })

    document.getElementById("msg").hidden = true;
//   document.getElementById("awesome").hidden = false;


    let elements = document.getElementsByClassName("predict");

    let clickHandler = function(event) {
        classify(event.currentTarget, true);
    };

    for (var i = 0; i < elements.length; i++) {
        elements[i].addEventListener('click', clickHandler, false);
    }
}


main();
