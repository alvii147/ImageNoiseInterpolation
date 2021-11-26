<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">openImage</span>(<span style="color: darkblue;">path</span>)</span></span>

<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">saveImage</span>(<span style="color: darkblue;">img</span>, <span style="color: darkblue;">path</span>)</span></span>

<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">splitChannels</span>(<span style="color: darkblue;">img</span>)</span></span>

<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">combineChannels</span>(<span style="color: darkblue;">R</span>, <span style="color: darkblue;">G</span>, <span style="color: darkblue;">B</span>)</span></span>

<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">addNoise</span>(<span style="color: darkblue;">channel</span>, <span style="color: darkblue;">p</span>=<span style="color: red;">0.5</span>, <span style="color: darkblue;">is_noisy</span>=<span style="color: red;">None</span>)</span></span>

<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">slidingWindowOperation</span>(<span style="color: darkblue;">channel</span>, <span style="color: darkblue;">window_shape</span>, <span style="color: darkblue;">op</span>=<span style="color: red;">'mean'</span>)</span></span>

<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">detectNoise</span>(<span style="color: darkblue;">channel</span>, <span style="color: darkblue;">N</span>, <span style="color: darkblue;">E</span>=<span style="color: red;">53</span>)</span></span>

<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">getCandidate</span>(<span style="color: darkblue;">C</span>, <span style="color: darkblue;">A</span>, <span style="color: darkblue;">idx</span>)</span></span>

<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">getValidity</span>(<span style="color: darkblue;">is_noisy_C</span>, <span style="color: darkblue;">is_noisy_A</span>, <span style="color: darkblue;">idx</span>)</span></span>

<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">getDeference</span>(<span style="color: darkblue;">C</span>, <span style="color: darkblue;">A</span>, <span style="color: darkblue;">idx</span>)</span></span>

<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">interpolatePixel</span>(<span style="color: darkblue;">C</span>, <span style="color: darkblue;">A1</span>, <span style="color: darkblue;">A2</span>, <span style="color: darkblue;">is_noisy_C</span>, <span style="color: darkblue;">is_noisy_A1</span>, <span style="color: darkblue;">is_noisy_A2</span>)</span></span>

<span style="font-family: monospace; font-weight: bold;"><span style="color: blue;">utils.</span><span style="font-size: 18pt;"><span style="color: orange;">interpolateChannel</span>(<span style="color: darkblue;">C</span>, <span style="color: darkblue;">A1</span>, <span style="color: darkblue;">A2</span>, <span style="color: darkblue;">is_noisy_C</span>, <span style="color: darkblue;">is_noisy_A1</span>, <span style="color: darkblue;">is_noisy_A2</span>)</span></span>

