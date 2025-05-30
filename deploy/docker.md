<!--
 * @Author: jhq
 * @Date: 2025-05-20 11:18:31
 * @LastEditTime: 2025-05-20 11:35:28
 * @Description: 
-->
* 拉镜像：
    ```bash
    docker pull ubuntu:22.04
* 启动镜像：
    docker run -it -v .:/home -p 9000:9000 ubuntu:22.04 /bin/bash
    docker run -it --rm --name fs-kb-server   -p 9000:9000 -v ./data:/home/fskb/fs-kb-server/data -v ./license:/home/fskb/fs-kb-server/license --dns 8.8.8.8  --network host  fs-kb-server:latest
* load镜像：
    docker load -i fs-kb-server.image.tar.gz

* 宿主机到容器复制文件
    docker cp fs-kb-server.tar.gz ae06665ef0a7:/home/jhq/fs-kb

* scp:
    scp wxg@192.168.1.242:/home/wxg/package/fs-kb-server.image.tar.gz ./