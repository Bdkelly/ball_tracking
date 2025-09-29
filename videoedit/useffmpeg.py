import os

def run_fmpeg(video_dir,video_name,vid_type,out_dir,fps):
    drop_dir = os.path.join(out_dir,video_name)
    outdir = drop_dir+"/imgs/"
    outframe = outdir+"frame_%05d.jpg"
    getvid = video_dir+video_name+vid_type
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ffcomd = (f'ffmpeg -i "{getvid}" -vf \"fps={fps}\" "{outframe}"')
    print (ffcomd)
    os.system(ffcomd)

if __name__ == "__main__":
    video_d = r"/Users/Ben/Documents/dever/python/ptorch/data/outframes/dips/p10/"
    video_n = "dips09_21_25_part_010"
    v_type = ".mp4"
    out_d = r"/Users/Ben/Documents/dever/python/ptorch/data/outframes/dips/p10"
    fps = "25.01"
    run_fmpeg(video_d,video_n,v_type,out_d,fps)