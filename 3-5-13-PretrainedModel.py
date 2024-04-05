#!/usr/bin/env python
# coding: utf-8

# using TF2.x with Keras 2.x see https://keras.io/getting_started/ and https://github.com/tensorflow/tensorflow/issues/63849
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"


# # Generate the Pretrained Model
# This notebook uses the pre-trained [micro_speech](https://github.com/tensorflow/tensorflow/tree/v2.4.1/tensorflow/lite/micro/examples/micro_speech) example for [TensorFlow Lite for MicroControllers](https://www.tensorflow.org/lite/microcontrollers/overview) 20 kB [Simple Audio Recognition](https://www.tensorflow.org/tutorials/sequences/audio_recognition) model to recognize keywords! **We strongly suggest you take your time working through this file to start to understand the code as we will be using a very similar file to train the model with your choice of keywords during the assignment.**

# ### Import packages
# Clone the TensorFlow Github Repository, which contains the relevant code required to run this tutorial. And import the old version of TF1 for backwards compatibility.


# get_ipython().system('wget https://github.com/tensorflow/tensorflow/archive/v2.14.0.zip')
# get_ipython().system('unzip v2.14.0.zip &> 0')
# get_ipython().system('mv tensorflow-2.14.0/ tensorflow/')


import tensorflow.compat.v1 as tf
import sys
# We add this path so we can import the speech processing modules.
sys.path.append("/files/pico/ML/tensorflow/tensorflow/examples/speech_commands/")
import input_data
import models
import numpy as np
import pickle


# ### Configure Defaults
# In this Colab we will just run with the default configurations to use the pre-trained model. However, in your assignment you will try the model to recognize a new word.


# A comma-delimited list of the words you want to train for.
# All the other words you do not select will be used to train
# an "unknown" label so that the model does not just recognize
# speech but your specific words. Audio data with no spoken
# words will be used to train a "silence" label.
WANTED_WORDS = "yes,no"

# Print the configuration to confirm it
print("Spotting these words: %s" % WANTED_WORDS)


# **DO NOT MODIFY** the following constants as they include filepaths used in this notebook and data that is shared during training and inference.


# Calculate the percentage of 'silence' and 'unknown' training samples required
# to ensure that we have equal number of samples for each label.
number_of_labels = WANTED_WORDS.count(',') + 1
number_of_total_labels = number_of_labels + 2 # for 'silence' and 'unknown' label
equal_percentage_of_training_samples = int(100.0/(number_of_total_labels))
SILENT_PERCENTAGE = equal_percentage_of_training_samples
UNKNOWN_PERCENTAGE = equal_percentage_of_training_samples

# Constants which are shared during training and inference
PREPROCESS = 'micro'
WINDOW_STRIDE = 20
MODEL_ARCHITECTURE = 'tiny_conv'

# Constants for training directories and filepaths
DATASET_DIR =  'dataset/'
LOGS_DIR = 'logs/'
TRAIN_DIR = 'train/' # for training checkpoints and other files.

# Constants for inference directories and filepaths
import os
MODELS_DIR = 'models'
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)
MODEL_TF = os.path.join(MODELS_DIR, 'model.pb')
MODEL_TFLITE = os.path.join(MODELS_DIR, 'model.tflite')
FLOAT_MODEL_TFLITE = os.path.join(MODELS_DIR, 'float_model.tflite')
MODEL_TFLITE_MICRO = os.path.join(MODELS_DIR, 'model.cc')
SAVED_MODEL = os.path.join(MODELS_DIR, 'saved_model')

# Constants for Quantization
QUANT_INPUT_MIN = 0.0
QUANT_INPUT_MAX = 26.0
QUANT_INPUT_RANGE = QUANT_INPUT_MAX - QUANT_INPUT_MIN

# Constants for audio process during Quantization and Evaluation
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 30.0
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0.8
BACKGROUND_VOLUME_RANGE = 0.1
TIME_SHIFT_MS = 100.0

# URL for the dataset and train/val/test split
DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
VALIDATION_PERCENTAGE = 10
TESTING_PERCENTAGE = 10


# ### Loading the pre-trained model
#
# These commands will download a pre-trained model checkpoint file (the output from training) that we can use to build a model.

# In[ ]:


get_ipython().system('curl -O "https://storage.googleapis.com/download.tensorflow.org/models/tflite/speech_micro_train_2020_05_10.tgz"')
get_ipython().system('tar xzf speech_micro_train_2020_05_10.tgz')
TOTAL_STEPS = 15000 # used to identify which checkpoint file


# ### Generate a TensorFlow Model for Inference
#
# Combine relevant training results (graph, weights, etc) into a single file for inference. This process is known as freezing a model and the resulting model is known as a frozen model/graph, as it cannot be further re-trained after this process.

# In[ ]:


get_ipython().system('rm -rf {SAVED_MODEL}')
get_ipython().system("python tensorflow/tensorflow/examples/speech_commands/freeze.py  --wanted_words=$WANTED_WORDS  --window_stride_ms=$WINDOW_STRIDE  --preprocess=$PREPROCESS  --model_architecture=$MODEL_ARCHITECTURE  --start_checkpoint=$TRAIN_DIR$MODEL_ARCHITECTURE'.ckpt-'{TOTAL_STEPS}  --save_format=saved_model  --output_file={SAVED_MODEL}")


# ### Generate a TensorFlow Lite Model
#
# Convert the frozen graph into a TensorFlow Lite model, which is fully quantized for use with embedded devices.
#
# The following cell will also print the model size, which will be under 20 kilobytes.
#
# We download the dataset to use as a representative dataset for more thoughtful post training quantization.
#
# **Note: this may take a little time as it is a relatively large file**

# In[ ]:


model_settings = models.prepare_model_settings(
    len(input_data.prepare_words_list(WANTED_WORDS.split(','))),
  SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
  WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
audio_processor = input_data.AudioProcessor(
    DATA_URL, DATASET_DIR,
  SILENT_PERCENTAGE, UNKNOWN_PERCENTAGE,
  WANTED_WORDS.split(','), VALIDATION_PERCENTAGE,
    TESTING_PERCENTAGE, model_settings, LOGS_DIR)


# In[ ]:


with tf.Session() as sess:
# with tf.compat.v1.Session() as sess: #replaces the above line for use with TF2.x
    float_converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    float_tflite_model = float_converter.convert()
    float_tflite_model_size = open(FLOAT_MODEL_TFLITE, "wb").write(float_tflite_model)
    print("Float model is %d bytes" % float_tflite_model_size)

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.lite.constants.INT8
    # converter.inference_input_type = tf.compat.v1.lite.constants.INT8 #replaces the above line for use with TF2.x
    converter.inference_output_type = tf.lite.constants.INT8
    # converter.inference_output_type = tf.compat.v1.lite.constants.INT8 #replaces the above line for use with TF2.x
    def representative_dataset_gen():
        for i in range(100):
            data, _ = audio_processor.get_data(1, i*1, model_settings,
                                         BACKGROUND_FREQUENCY,
                                         BACKGROUND_VOLUME_RANGE,
                                         TIME_SHIFT_MS,
                                         'testing',
                                         sess)
            flattened_data = np.array(data.flatten(), dtype=np.float32).reshape(1, 1960)
            yield [flattened_data]
    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    tflite_model_size = open(MODEL_TFLITE, "wb").write(tflite_model)
    print("Quantized model is %d bytes" % tflite_model_size)


# ### Testing the accuracy after Quantization
#
# Verify that the model we've exported is still accurate, using the TF Lite Python API and our test set.

# In[ ]:


# Helper function to run inference
def run_tflite_inference_testSet(tflite_model_path, model_type="Float"):
    #
    # Load test data
    #
    np.random.seed(0) # set random seed for reproducible test results.
    with tf.Session() as sess:
    # with tf.compat.v1.Session() as sess: #replaces the above line for use with TF2.x
        test_data, test_labels = audio_processor.get_data(
        -1, 0, model_settings, BACKGROUND_FREQUENCY, BACKGROUND_VOLUME_RANGE,
      TIME_SHIFT_MS, 'testing', sess)
    test_data = np.expand_dims(test_data, axis=1).astype(np.float32)

    #
    # Initialize the interpreter
    #
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    #
    # For quantized models, manually quantize the input data from float to integer
    #
    if model_type == "Quantized":
        input_scale, input_zero_point = input_details["quantization"]
        test_data = test_data / input_scale + input_zero_point
        test_data = test_data.astype(input_details["dtype"])

    #
    # Evaluate the predictions
    #
    correct_predictions = 0
    for i in range(len(test_data)):
        interpreter.set_tensor(input_details["index"], test_data[i])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        top_prediction = output.argmax()
        correct_predictions += (top_prediction == test_labels[i])

    print('%s model accuracy is %f%% (Number of test samples=%d)' % (
      model_type, (correct_predictions * 100) / len(test_data), len(test_data)))


# In[ ]:


# Compute float model accuracy
run_tflite_inference_testSet(FLOAT_MODEL_TFLITE)

# Compute quantized model accuracy
run_tflite_inference_testSet(MODEL_TFLITE, model_type='Quantized')


# # Testing the model on example Audio
# Now that we know the model is accurate on the test set lets explore with some hand crafted examples just how accurate the model is in the real world!

# ### Load and listen to the example files
# What is interesting about them? Can you tell them all apart?

# In[ ]:


from IPython.display import HTML, Audio
get_ipython().system('wget --no-check-certificate --content-disposition https://github.com/tinyMLx/colabs/blob/master/yes_no.pkl?raw=true')
print("Wait a minute for the file to sync in the Colab and then run the next cell!")


# In[ ]:


fid = open('yes_no.pkl', 'rb')
audio_files = pickle.load(fid)
yes1 = audio_files['yes1']
yes2 = audio_files['yes2']
yes3 = audio_files['yes3']
yes4 = audio_files['yes4']
no1 = audio_files['no1']
no2 = audio_files['no2']
no3 = audio_files['no3']
no4 = audio_files['no4']
sr_yes1 = audio_files['sr_yes1']
sr_yes2 = audio_files['sr_yes2']
sr_yes3 = audio_files['sr_yes3']
sr_yes4 = audio_files['sr_yes4']
sr_no1 = audio_files['sr_no1']
sr_no2 = audio_files['sr_no2']
sr_no3 = audio_files['sr_no3']
sr_no4 = audio_files['sr_no4']


# In[ ]:


Audio(yes1, rate=sr_yes1)


# In[ ]:


Audio(yes2, rate=sr_yes2)


# In[ ]:


Audio(yes3, rate=sr_yes3)


# In[ ]:


Audio(yes4, rate=sr_yes4)


# In[ ]:


Audio(no1, rate=sr_no1)


# In[ ]:


Audio(no2, rate=sr_no2)


# In[ ]:


Audio(no3, rate=sr_no3)


# In[ ]:


Audio(no4, rate=sr_no4)


# ### Test the model on the example files
# We first need to import a series of packages and build the loudest section tool so that we can process audio files manually to send them to our model. These packages will also be used later for you to record your own audio to test the model!

# In[ ]:


get_ipython().system('pip install ffmpeg-python &> 0')
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg
get_ipython().system('pip install librosa')
import librosa
import scipy.io.wavfile
get_ipython().system('git clone https://github.com/petewarden/extract_loudest_section.git')
get_ipython().system('make -C extract_loudest_section/')
print("Packages Imported, Extract_Loudest_Section Built")


# In[ ]:


# Helper function to run inference (on a single input this time)
# Note: this also includes additional manual pre-processing
TF_SESS = tf.compat.v1.InteractiveSession()
def run_tflite_inference_singleFile(tflite_model_path, custom_audio, sr_custom_audio, model_type="Float"):
    #
    # Preprocess the sample to get the features we pass to the model
    #
    # First re-sample to the needed rate (and convert to mono if needed)
    custom_audio_resampled = librosa.resample(librosa.to_mono(np.float64(custom_audio)), orig_sr = sr_custom_audio, target_sr = SAMPLE_RATE)
    # Then extract the loudest one second
    scipy.io.wavfile.write('custom_audio.wav', SAMPLE_RATE, np.int16(custom_audio_resampled))
    get_ipython().system('/tmp/extract_loudest_section/gen/bin/extract_loudest_section custom_audio.wav ./trimmed')
    # Finally pass it through the TFLiteMicro preprocessor to produce the
    # spectrogram/MFCC input that the model expects
    custom_model_settings = models.prepare_model_settings(
      0, SAMPLE_RATE, CLIP_DURATION_MS, WINDOW_SIZE_MS,
    WINDOW_STRIDE, FEATURE_BIN_COUNT, PREPROCESS)
    custom_audio_processor = input_data.AudioProcessor(None, None, 0, 0, '', 0, 0,
                                                     model_settings, None)
    custom_audio_preprocessed = custom_audio_processor.get_features_for_wav(
      'trimmed/custom_audio.wav', model_settings, TF_SESS)
    # Reshape the output into a 1,1960 matrix as that is what the model expects
    custom_audio_input = custom_audio_preprocessed[0].flatten()
    test_data = np.reshape(custom_audio_input,(1,len(custom_audio_input)))

    #
    # Initialize the interpreter
    #
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    #
    # For quantized models, manually quantize the input data from float to integer
    #
    if model_type == "Quantized":
        input_scale, input_zero_point = input_details["quantization"]
        test_data = test_data / input_scale + input_zero_point
        test_data = test_data.astype(input_details["dtype"])

    #
    # Run the interpreter
    #
    interpreter.set_tensor(input_details["index"], test_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    top_prediction = output.argmax()

    #
    # Translate the output
    #
    top_prediction_str = ''
    if top_prediction == 2 or top_prediction == 3:
        top_prediction_str = WANTED_WORDS.split(',')[top_prediction-2]
    elif top_prediction == 0:
        top_prediction_str = 'silence'
    else:
        top_prediction_str = 'unknown'

    print('%s model guessed the value to be %s' % (model_type, top_prediction_str))


# In[ ]:


# Then test the model -- do they all work as you'd expect?
print("Testing yes1")
run_tflite_inference_singleFile(MODEL_TFLITE, yes1, sr_yes1, model_type="Quantized")
print("Testing yes2")
run_tflite_inference_singleFile(MODEL_TFLITE, yes2, sr_yes2, model_type="Quantized")
print("Testing yes3")
run_tflite_inference_singleFile(MODEL_TFLITE, yes3, sr_yes3, model_type="Quantized")
print("Testing yes4")
run_tflite_inference_singleFile(MODEL_TFLITE, yes4, sr_yes4, model_type="Quantized")
print("Testing no1")
run_tflite_inference_singleFile(MODEL_TFLITE, no1, sr_no1, model_type="Quantized")
print("Testing no2")
run_tflite_inference_singleFile(MODEL_TFLITE, no2, sr_no2, model_type="Quantized")
print("Testing no3")
run_tflite_inference_singleFile(MODEL_TFLITE, no3, sr_no3, model_type="Quantized")
print("Testing no4")
run_tflite_inference_singleFile(MODEL_TFLITE, no4, sr_no4, model_type="Quantized")


# # Testing the model with your own data!

# ### Define the audio importing function
# Adapted from: https://ricardodeazambuja.com/deep_learning/2019/03/09/audio_and_video_google_colab/ and https://colab.research.google.com/drive/1Z6VIRZ_sX314hyev3Gm5gBqvm1wQVo-a#scrollTo=RtMcXr3o6gxN

# In[ ]:


AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_btn.appendChild(t);
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  gumStream = stream;
  var options = {
    bitsPerSecond: 128000, //chrome seems to ignore, always 48k
    audioBitsPerSecond: 128000, //chrome seems to ignore, always 48k
    mimeType : 'audio/mp4'
    // mimeType : 'audio/webm;codecs=opus' // try me if the above fails
  };
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {
    var url = URL.createObjectURL(e.data);
    var preview = document.createElement('audio');
    preview.controls = true;
    preview.src = url;
    document.body.appendChild(preview);

    reader = new FileReader();
    reader.readAsDataURL(e.data);
    reader.onloadend = function() {
      base64data = reader.result;
    }
  };
  recorder.start();
  };

recordButton.innerText = "Recording... press to stop";

navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);


function toggleRecording() {
  if (recorder && recorder.state == "recording") {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Saving the recording... pls wait!"
  }
}

// https://stackoverflow.com/a/951057
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

var data = new Promise(resolve=>{
//recordButton.addEventListener("click", toggleRecording);
recordButton.onclick = ()=>{
toggleRecording()

sleep(2000).then(() => {
  // wait 2000ms for the data to be available...
  // ideally this should use something like await...
  resolve(base64data.toString())

});

}
});

</script>
"""

def get_audio():
    display(HTML(AUDIO_HTML))
    data = eval_js("data")
    binary = b64decode(data.split(',')[1])

    process = (ffmpeg
             .input('pipe:0')
             .output('pipe:1', format='wav', ac='1')
             .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
    )
    output, err = process.communicate(input=binary)

    riff_chunk_size = len(output) - 8
    # Break up the chunk size into four bytes, held in b.
    q = riff_chunk_size
    b = []
    for i in range(4):
        q, r = divmod(q, 256)
        b.append(r)

    # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
    riff = output[:4] + bytes(b) + output[8:]

    sr, audio = wav_read(io.BytesIO(riff))

    return audio, sr
print("Chrome Audio Recorder Defined")


# ### Record your own audio and test the model!
# After you run the record cell wait for the stop button to appear then start recording and then press the button to stop the recording once you have said the word!

# In[ ]:


custom_audio, sr_custom_audio = get_audio()
print("DONE")


# In[ ]:


# Then test the model
run_tflite_inference_singleFile(MODEL_TFLITE, custom_audio, sr_custom_audio, model_type="Quantized")

