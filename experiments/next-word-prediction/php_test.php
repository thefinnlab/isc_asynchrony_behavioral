<?php

header("Cross-Origin-Opener-Policy: same-origin");
header("Cross-Origin-Embedder-Policy: credentialless");
// header("Cross-Origin-Resource-Policy: cross-origin");
// header("Access-Control-Allow-Origin: *");

include('simple_video_player.html');
?>

<!-- <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
<script src="https://unpkg.com/jspsych@7.2.3"></script>
<script src="https://unpkg.com/@jspsych/plugin-html-keyboard-response@1.1.1"></script>
<script src="https://unpkg.com/@jspsych/plugin-fullscreen@1.1.1"></script>
<script src="https://unpkg.com/@jspsych/plugin-survey-text@1.1.1"></script>
<script src="https://unpkg.com/@jspsych/plugin-call-function@1.1.1"></script>
<script src="https://unpkg.com/@jspsych/plugin-instructions@1.1.1"></script>
<style>
  body {
    background-color: black;
    font-family: sans-serif;
    color: white;
  }
  canvas {
    margin: 0 auto;
  }
  #controls {
    margin-bottom: 10px;
  }
  #loading {
    font-size: 2em;
  }
</style>
<div id=controls>
  <div id=loading>Loading...</div>
  <button disabled=true>Play</button>
  <label for=volume>Volume</label>
  <input id=volume type=range value=0.8 min=0 max=1.0 step=0.01></input>
  Audio buffer health:
  <progress min=0 max=100></progress>
  Total output latency:
  <span id=totalOutputLatency>N/A</span>
  <input id=outputLatency type=checkbox></input>
  Use AudioContext.outputLatency (Connect Bluetooth headset for best results)
</div>
<canvas style="outline: 1px solid"></canvas>

<script src="mp4box.all.min.js"></script>

<script type="module">
  window.$ = document.querySelector.bind(document);
  import {VideoRenderer} from "./video_renderer.js";
  import {AudioRenderer} from "./audio_renderer.js";

  let audioRenderer = new AudioRenderer();
  let audioReady = audioRenderer.initialize('bbb_audio_aac_frag.mp4');

  // TODO: move to worker. Use OffscreenCanvas.
  let canvas = $("canvas");
  let videoRenderer = new VideoRenderer();
  let videoReady = videoRenderer.initialize('bbb_video_avc_frag.mp4', canvas);

  await Promise.all([audioReady, videoReady]);

  let totalOutputLatencyElement = $('#totalOutputLatency');
  let isAudioContextOutputLatencySupported = ('outputLatency' in AudioContext.prototype);
  let useAudioContextOutputLatency = isAudioContextOutputLatencySupported;
  $('#outputLatency').disabled = !isAudioContextOutputLatencySupported;
  $('#outputLatency').checked = isAudioContextOutputLatencySupported;
  $('#outputLatency').onchange = function(e) {
    useAudioContextOutputLatency = e.target.checked;
  }

  $('#volume').onchange = function(e) {
    audioRenderer.setVolume(e.target.value);
  }

  let playing = false;
  let progressElement = $('progress');
  progressElement.value = audioRenderer.bufferHealth();

  let playButton = $('button');
  let loadingElement = $('#loading');

  // Enable play now that we're loaded
  playButton.disabled = false;
  loadingElement.innerText = 'Ready! Click play.'

  playButton.onclick = function(e) {
    if (playButton.innerText == "Play") {
      console.log("playback start");
      playing = true;
      // Audio can only start in reaction to a user-gesture.
      audioRenderer.play().then(() => console.log("playback started"));
      playButton.innerText = "Pause";

      window.requestAnimationFrame(function renderVideo() {
        if (!playing)
          return;

        let totalOutputLatency = audioRenderer.getTotalOutputLatencyInSeconds(useAudioContextOutputLatency);
        totalOutputLatencyElement.textContent = `${Math.round(totalOutputLatency * 1000)}ms`;
        videoRenderer.render(audioRenderer.getMediaTimeInMicroSeconds(totalOutputLatency));
        progressElement.value = audioRenderer.bufferHealth();
        window.requestAnimationFrame(renderVideo);
      });
    } else {
      console.log("playback pause");
      playing = false;
      audioRenderer.pause().then(() => console.log("playback pause"));
      playButton.innerText = "Play"
    }
  }
</script>

</html> -->