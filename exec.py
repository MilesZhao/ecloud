import os

for ep in [100,125,175,200]:
    for lr in [0.001,0.0005]:
        os.system("python3 fusionnet.py --num_epochs={} --learning_rate={}".format(ep, lr))



