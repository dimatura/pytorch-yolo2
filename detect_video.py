import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import av
from spath import Path

def detect(cfgfile, weightfile, videofile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    # img = Image.open(imgfile).convert('RGB')
    class_names = load_class_names(namesfile)

    video = av.open(videofile)
    stream = next(s for s in video.streams if s.type == 'video')

    outvideo = av.open(str('predictions.mp4'), 'w')
    # outstream = outvideo.add_stream('mpeg4', stream.rate)
    # outstream = outvideo.add_stream('h264', '23.976')
    # outstream = outvideo.add_stream('libx264', '23.976')
    outstream = outvideo.add_stream('mpeg4', '24')
    # outstream.bit_rate = 640000000
    # outstream.bit_rate = 60033851
    outstream.bit_rate = 8000000
    # outstream.crf = '19'
    outstream.pix_fmt = 'yuv420p'
    # outstream.height = int(m.height)
    # outstream.width = int(m.width)
    outstream.height = int(720)
    outstream.width = int(1080)

    outdir = Path('./frames/').makedirs_p()

    frame_ctr = 0

    for packet in video.demux(stream):
        for frame in packet.decode():
            img = frame.to_image().convert('RGB')
            sized = img.resize((m.width, m.height), Image.BILINEAR)

            start = time.time()
            boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            finish = time.time()
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

            pred_img = plot_boxes(img, boxes, None, class_names)
            pred_img = pred_img.resize((1024, 768), Image.BILINEAR)

            # outframe_fname = (outdir/'frame%05d.jpg'%frame_ctr)
            # pred_img.save(outframe_fname)

            # pred_img = pred_img.resize((640, 480))
            outframe = av.VideoFrame.from_image(pred_img)
            outpacket = outstream.encode(outframe)
            outvideo.mux(outpacket)

            frame_ctr += 1

    while True:
        outpacket = outstream.encode()
        if outpacket is not None:
            outvideo.mux(outpacket)
        else:
            break
    outvideo.close()

if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        detect(cfgfile, weightfile, imgfile)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
