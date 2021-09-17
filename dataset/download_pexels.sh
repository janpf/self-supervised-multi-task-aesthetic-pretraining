cat urls.txt | sed -e 's/$/?w=1000/' | tqdm | xargs -I {} sh -c 'cd /datasets/pexels/images; curl -C - -OJs {}'
