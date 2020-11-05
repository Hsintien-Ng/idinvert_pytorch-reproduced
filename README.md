# idinvert_pytorch-reproduced
 Our project is an edited version of https://github.com/genforce/idinvert_pytorch

 In this project, we reproduce the encoder training according to the tf version (https://github.com/genforce/idinvert) with celeba-hq dataset. We add a discriminator including models/base_discriminator.py, models/stylegan_discriminator_network.py, models/stylegan_discriminator.py, and the encoder training including training/training_loop_encoder.py, train_encoder.py.

 In order to achieve a parallel training, we also edit some lines in models/base_module.py, models/base_generator.py, models/base_encoder.py.

 Since I reproduced it based on my own knowledge, there may be some areas where my understanding is not in place. Welcome everyone to correct me.
