## Data Format
Here we give an overview of the data structure of each multimodal dataset used in our experiments. 

### Music AVQA
The structure should be:
```
root
├── video1.mp3
├── video1.mp4
├── video2.mp3
├── video2.mp4
...
```
### Valor
The structure should be:
```
root
├── video1
│   ├── video1.mp3
│   └── video1.mp4
├── video2
│   ├── video2.mp3
│   └── video2.mp4
│   ...
```
### CharadesEGO
The structure should be:
```
root
├── video
│   ├── video1.mp4
│   ├── video2.mp4
│    ...
├── audio
│   ├── audio1.mp3
│   ├── audio2.mp3
│    ...
```
### MOSI / MOSEI
The structure should be:
```
root
├── Raw
│   ├── video1
│   │   ├── clip1.mp4
│   │   ├── clip2.mp4
│   │   ...
│   ├── video2
│   │   ├── clip1.mp4
│   │   ├── clip2.mp4
│   │   ...
│    ...
├── Raw_audio
│   ├── video1
│   │   ├── clip1.mp3
│   │   ├── clip2.mp3
│   │   ...
│   ├── video2
│   │   ├── clip1.mp4
│   │   ├── clip2.mp3
│   │   ...
│    ...
└── labels.csv

```