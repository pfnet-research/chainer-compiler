- For CI

```bash
$ docker build --target ci-base -t TAG -f docker/Dockerfile .
```

- For base of developement
  - include built chainerx and chainer-compiler

```bash
$ docker build --target dev-base -t TAG -f docker/Dockerfile .
```

- For developement with tools

```bash
$ docker build -t TAG -f docker/Dockerfile .
```
