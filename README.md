# The MIDI Wand: A Gesture-Based MIDI Controller Using Deep Learning
GitHub Repo: https://github.com/JoeBourd/CASA0018_MIDI_Wand
Edge Impulse project: https://studio.edgeimpulse.com/public/968120/live

## Introduction
The digitisation of music production has democratised it, with people able to craft albums in their bedrooms. Human computer interaction (HCI) is now central to digital musical instrument design, but often this interface leads to a loss of tangibility and organic interactions that arise with their acoustic counterparts (Morrison and McPherson, 2024; Overholt, 2012). Virtual instruments that emulate or sample acoustic sounds are often controlled by generic faders, knobs, and mod wheels built into standard MIDI controllers. This fixes the sensing mode of touch to a single plane, restricted within a relatively small range of motion, reducing the expressive capability of the user. It also constrains the relationship between the player and the instrument to one that is unidirectional: data flows from the controller to the parameter (Davison and McPherson, 2026). Acoustic instruments enable a sense of bidirectionality, where the mechanical interactions between the player and the instrument can influence how the instrument is played, becoming of an extension of the player.

Gesture sensing instruments are increasingly capable of capturing dynamic human movement, ultimately acting as a conduit that enables the hand itself to become an instrument. A key example is the MiMU glove, which uses accelerometers and gravimeters to capture gestures that can be mapped to digital instruments for parameter control (Kanga, 2023). New gestures are learned through the Glover software, MiMU’s bespoke integrating machine learning tool. Arguably this learning capability leans the device towards being a bidirectional interface, since players may adapt their gestures to mimic how the sonic response ‘feels’, in an iterative process of gesture development and machine learning. 

The MIDI Wand is inspired by these principles of bringing organic interactions into digital music creation, providing a framework that uses deep learning to interpret gestures into MIDI outputs. It transforms the actions in moving faders and knobs into more expressive, naturalistic movements, aiming to augment player responses. The use of embedded AI here allows the device to be held and manipulated, with processing happening on the device before being sent to actuate a MIDI change.

## Application Overview
The MIDI Wand aims to capture gestures while it sits the user’s hand through the Arduino Nano 33 BLE Sense’s integrated accelerometer, gyroscope, and magnetometer. These data are fed into a TensorFlow Lite model, which will predict the gesture from one of nine classes. These are split into two sets of broader gestures that correspond to a fader and a knob movement.

The classes are: FaderUpFast, FaderUpSlow, FaderDownFast, FaderDownSlow, KnobUpFast, KnobUpSlow, KnobDownFast, KnobDownSlow, Unknown

Once inference is complete, the predicted class will action a specific MIDI control change. The fader and knob gestures will be grouped and assigned one MIDI continuous controller (CC) each, so they can be mapped to a single digital parameter such as a virtual knob or fader. The amounts of increase and decrease will be constant as default, with only the direction and speed of outputs changing across classes. These MIDI control changes will be sent via USB to a computer, where it can be mapped to parameters in, for example, a digital audio workstation (DAW).



<img width="211" height="130" alt="image" src="https://github.com/user-attachments/assets/3525ee25-83a2-4cb9-93e7-153ab728fb29" />


## Data Collection
Data were collected using the same Arduino Nano 33 BLE Sense directly into Edge Impulse via WebUSB. To create a rough spatial framework for the gestures, two points were marked on a wall, arranged vertically at a distance that felt natural to move through. This also enabled the introduction of variation around the default movement to mimic natural variation in hand movements. Sampling begun with the hand at a relaxed position by the side of the body, then brought up to the bottom marker. The Arduino was held by its USB connector to ensure consistent orientation. For fader movements, the Arduino was moved up or down between markers at different paces corresponding to the gesture class, while for knob movements the Arduino was rotated 90° about the bottom marker. 


<img width="220" height="185" alt="Screenshot 2026-05-03 at 11 36 22" src="https://github.com/user-attachments/assets/244d8278-41f8-498b-a229-ce0d448fecd0" />


<img width="233" height="161" alt="Screenshot 2026-05-03 at 11 37 57" src="https://github.com/user-attachments/assets/48f94ef8-41a7-404f-ad0c-3bf056c89c2e" />


Each gesture was sampled continuously for around ten minutes and manually split into discrete gesture events. ‘Fast’ segments were around 2000ms, while ‘slow’ segments were around 4000ms, and ‘unknown’ segments were set at 2500ms. Overall, 1h 30m 31s data was collected. 20% of these samples were moved to the test dataset to achieve an 80/20 training/test split.



<img width="513" height="295" alt="Screenshot 2026-05-03 at 11 38 26" src="https://github.com/user-attachments/assets/14dbfc11-a880-4a0f-9bdd-038eb1cf4b46" />


## Model
For the initial digital signal processing (DSP), it was thought that the spectral analysis processing block on Edge Impulse would work best. This is because it can extract key features from simple, low frequency movements using filtering (Edge Impulse Documentation, 2026). A convolutional neural network (CNN) architecture was hypothesised as most appropriate for this application due to its ability to learn relations between datapoints in a time series (Warden and Situnayake, 2019). This will be useful for learning features that involve similar directional movements but at different speeds. The CNN can extract specific features within the larger movement, which are then fed into dense layers which will enable the model to learn which combinations of features constitute a certain gesture (Warden and Situnayake, 2019).
 


<img width="430" height="252" alt="Screenshot 2026-05-03 at 11 38 45" src="https://github.com/user-attachments/assets/7fb91143-ca75-4362-9262-01254da6bee1" />



## Experiments
Experiments were split into two stages: choosing the DSP and feature extraction parameters and choosing the neural network architecture. DSP experiments were carried out while holding the neural architecture at its default settings, which were two dense layers only. The Syntiant IMU and spectral analysis processing blocks were tested, with the spectral analysis block having more adjustable parameters. Adding a low pass filter at 3Hz to the spectral analysis block improved accuracy, likely due to filtering out unwanted high frequency noise that was not part of the main gesture. This was supported by the increased visual similarity between samples post processing. Increasing the FFT length increases resolution of the frequency peaks, but there were limited returns after a length of 32. Window size had a significant impact on accuracy here, with a maximum of 5000ms performing the best, possibly as it captures the full length of the longest gesture. To reduce latency at inferencing, however, smaller window sizes were trialled and 2500ms was found to be the best trade-off between sampling duration and accuracy.


<img width="430" height="187" alt="Screenshot 2026-05-03 at 11 39 04" src="https://github.com/user-attachments/assets/a984cb22-798d-493c-a809-a1c50d089def" />


<img width="512" height="201" alt="Screenshot 2026-05-03 at 11 39 49" src="https://github.com/user-attachments/assets/66f16886-f397-4a30-a98a-f3a5a4eb838d" />

<img width="407" height="251" alt="Screenshot 2026-05-03 at 11 40 03" src="https://github.com/user-attachments/assets/5e8e8d09-0174-4002-adbb-d771613aa848" />


The optimal run from these experiments (run 13) was used to then test the neural network architecture. The architecture was initialised based on TinyML’s ‘Magic Wand’ example, which was trained to pick out discrete gestures (Warden and Situnayake, 2019). Larger filter and kernel sizes increased accuracy in the convolutional layers, and an optimum dense layer neuron size was found as 50 and 20 respectively. Learning rate was increased from 0.0005 to 0.001 and 0.01, but the initial lower rate performed the best. This was also found with dropout rates between both layer types. To test whether reducing model size and therefore complexity would impact inferencing, one convolutional layer was removed, which appeared to increase training and validation accuracy. By reducing model complexity, this could prevent overfitting as well as reduce latency during inferencing (Warden and Situnayake, 2019).



<img width="517" height="244" alt="Screenshot 2026-05-03 at 11 40 42" src="https://github.com/user-attachments/assets/52b3c029-4cc6-4a89-8be9-a422822bb06b" />



During model testing, samples in the test dataset that were predicted with low accuracy were fed into the training dataset and the model was retrained. Live classification was also experimented with, and it was noticed that when the device was stationary, it would often misclassify as FaderDownSlow, so an extra minute of stationary data was added to the Unknown class and the model was retrained. The final architecture was exported as a quantised TensorFlow Lite model.



<img width="282" height="334" alt="Screenshot 2026-05-03 at 11 41 00" src="https://github.com/user-attachments/assets/a59ea179-e691-47e2-9ad4-392d1743b4b8" />




## Results and Observations
Across experiments, validation accuracy increased from 94.6% to 98.6%, while validation loss decreased from 0.37 to 0.07. Final test accuracy was 97%. The confusion matrix shows that the model only predicts Unknown gestures 65.6% of the time, with the most wrong predictions attributed to FaderUpFast and KnobDownSlow. This will likely lead to unwanted parameter changes when using the application. The loss graph shows convergence happening around 20 epochs, with the validation loss increasing again slightly afterwards. This suggests that the 50-epoch training length chosen may be unnecessary, though Edge Impulse will use the model parameters from the best performing epoch (Edge Impulse Documentation, 2026). The divergence seen in the last epochs is unlikely to reflect overfitting in the model as the difference in values are small, and the test accuracy is also not much lower than the validation accuracy (Warden and Situnayake, 2019).





<img width="492" height="273" alt="Screenshot 2026-05-03 at 11 41 22" src="https://github.com/user-attachments/assets/50435879-4f19-470d-b5de-07559fa5bc3a" />





Despite the model’s high accuracy, testing the MIDI Wand on a virtual instrument revealed that some gestures were not recognised consistently. This was especially true for fader movements, possibly because the model had learned similar up or down gestures as sampling had captured bringing the sensor to its initial start height or moving it to a relaxed position as well as the focal gesture. Recognising the start and end of a gesture is a key issue involved in sensor based gesture recognition, and better data segmentation at the data collection phase could mitigate this (Tai et al., 2018). Introducing more variation into the dataset also help, which may involve increasing the number of people to input data, as well as changing whether that person is standing or sitting. Currently, the output processing in Arduino IDE involves taking a predicted class above a 0.6 probability threshold and using it to action a series of stepwise MIDI outputs that increase or decrease a parameter over time. Sampling happens automatically once an output is finished. This limits the user’s ability to engage with the device expressively, as they can only action discrete outputs. They are also restricted temporally by the sampling buffer, so must wait when the device is ready to change a parameter. A continuous gesture model would enable more real time, precise outputs according to user input, and could be deployed on an edge device (Tai et al., 2018). Adding a button that could be pressed to start sampling would allow the user to control when they want a parameter changed. Adding an ergonomic housing could also promote more naturalistic movements and increase expressivity (Graf and Barthet, 2022).


## Conclusion
The MIDI Wand provides a window into using embedded machine learning to introduce embodied, expressive interaction into digital musical instrument design. By translating natural movement into a MIDI control change, it offers a more intuitive and dynamic interface than standard MIDI controllers. Although the model achieves high accuracy, tweaks at each stage of the development pipeline are necessary to fully realise its goal. While at the prototype stage, it demonstrates the potential for edge AI to augment HCI in music production.


## Bibliography
Davison, M., McPherson, A., 2026. Design Explorations of Instruments and Interactions with Bidirectional Haptic Couplings, in: Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems, CHI ’26. Association for Computing Machinery, New York, NY, USA, pp. 1–15. https://doi.org/10.1145/3772318.3791190
Edge Impulse Documentation [WWW Document], 2026. . Edge Impulse Documentation. URL https://docs.edgeimpulse.com/studio/projects/processing-blocks/blocks/spectral-analysis (accessed 5.1.26).
Graf, M., Barthet, M., 2022. Mixed Reality Musical Interface: Exploring Ergonomics and Adaptive Hand Pose Recognition for Gestural Control. International Conference on New Interfaces for Musical Expression. https://doi.org/10.21428/92fbeb44.56ba9b93
Kanga, Z., 2023. The Cyborg Hand: Gesture, Technology, Disability and Interdisciplinarity in Whatever Weighs You Down. Contemporary Music Review 42, 319–338. https://doi.org/10.1080/07494467.2023.2279357
Morrison, L., McPherson, A., 2024. Entangling Entanglement: A Diffractive Dialogue on HCI and Musical Interactions, in: Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, CHI ’24. Association for Computing Machinery, New York, NY, USA, pp. 1–17. https://doi.org/10.1145/3613904.3642171
Overholt, D., 2012. Violin-Related HCI: A Taxonomy Elicited by the Musical Interface Technology Design Space, in: Brooks, A.L. (Ed.), Arts and Technology. Springer, Berlin, Heidelberg, pp. 80–89. https://doi.org/10.1007/978-3-642-33329-3_10
Tai, T.-M., Jhang, Y.-J., Liao, Z.-W., Teng, K.-C., Hwang, W.-J., 2018. Sensor-Based Continuous Hand Gesture Recognition by Long Short-Term Memory. IEEE Sensors Letters 2, 1–4. https://doi.org/10.1109/LSENS.2018.2864963
Warden, P., Situnayake, D., 2019. TinyML. O’Rielly Media, Inc.



----

## Declaration of Authorship

I, Joe Bourdier, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.


*Joe Bourdier*

3/05/26

Word count: 1751
