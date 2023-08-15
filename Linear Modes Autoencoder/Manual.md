Below is the process of how to run the linear modes algorithm.

1. Generate training data via the "2DOF Mass Spring Data file", "3DOF Mass Spring Data file" or your own file. If using your own data please change the import/export format accordingly. Make sure to rename the filepath variable from "Input Filepath" to a desire filepath on your device.

2. Open either the linear modes autoencoder file best suited for your dataset

3. Inside the linear modes script, change the import filepath similar to point 1 such that the NN loads the correct training dataset. Again change the export modal response filepath (export) to a path of your liking. 

4. Run the linear modes autoencoder and chose the modes that make sense and there could be the presence of false modes.

5. Change the import filepath of the FFT file and run it to obatin the frequency vs amplitude representation of the modal response.
   
