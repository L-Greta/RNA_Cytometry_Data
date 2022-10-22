#!/usr/bin/env python3

VERSION = "0.1"
import json, sys, os, uuid, csv, copy

import numpy as np
import numpy.ma as ma
print("Numpy imported successfully, version: "+np.__version__, file=sys.stderr)

import cv2
print("Open CV imported successfully, version: "+cv2.__version__, file=sys.stderr)

import pandas as pd
print("Pandas imported successfully, version: "+pd.__version__, file=sys.stderr)


millis = lambda: int(round(time.time() * 1000))
print("Droplet Cytometry v."+VERSION+" is starting at: "+sys.path[0], file=sys.stderr)

##################################
## Load the configuration files
##################################
try:
    with open(sys.path[0]+'/data_structure.json') as json_file:
        data_structure = json.load(json_file)
except Exception as ex:
    print("Error parsing json files: ", str(ex), file=sys.stderr)
    sys.exit(0)

if(data_structure is None):
    print("Could not read data_structure", file=sys.stderr)
    sys.exit(0)

##################################
## Parse the configuration file
##################################
try:
    INPUT_DIR   = sys.path[0]+"/"+data_structure["input_dir"]
    OUTPUT_DIR  = sys.path[0]+"/"+data_structure["output_dir"]
except Exception as ex:
    print("Could not read configuration: ", str(ex), file=sys.stderr)
    sys.exit(0)

CROP_MARGIN = 5
CROP_SIZE = 128

##################################
## The main analysis framework structure
##################################
class Framework():

    # Initialize the class
    def __init__(self):
        "Inner layer communications class for aggregation and translation"
        self.data_main = copy.deepcopy(data_structure)

        self.batch_size = 0
        self.scale = 1
        self.results = dict()
        self.results["measurements"] = list()
        self.image_dir = OUTPUT_DIR+"/images"
        self.result_dir = OUTPUT_DIR+"/results"
        

        for ch in self.data_main["channels"]:
            self.data_main["channels"][ch]["path"] = INPUT_DIR+ch
            self.data_main["channels"][ch]["filepaths"] = [INPUT_DIR+ch+"/"+name for name in sorted(os.listdir( INPUT_DIR+ch )) if ("tif" in name) or ("png" in name) or ("jpg" in name) or ("jpeg" in name)]
            self.data_main["channels"][ch]["filenames"] = [name for name in sorted(os.listdir( INPUT_DIR+ch )) if ("tif" in name) or ("png" in name) or ("jpg" in name) or ("jpeg" in name)]
            # print(self.data_main["channels"][ch]["files"][0])
            self.batch_size = len(self.data_main["channels"][ch]["filenames"])
            if "scale" in list(self.data_main["channels"][ch].keys()):
                self.scale = float(self.data_main["channels"][ch]["scale"])

        if not os.path.exists( OUTPUT_DIR ):
            os.makedirs( OUTPUT_DIR )

        if not os.path.exists( self.image_dir ):
            os.makedirs( self.image_dir )

        if not os.path.exists( self.result_dir ):
            os.makedirs( self.result_dir )

        if not os.path.exists( os.path.join(self.result_dir, "crops") ):
            os.makedirs( os.path.join(self.result_dir, "crops") )

        if not os.path.exists( os.path.join(self.result_dir, "crops_orig") ):
            os.makedirs( os.path.join(self.result_dir, "crops_orig") )


    # Load brightfield image
    def load_brightfield(self, idx):
        for ch in self.data_main["channels"]:
            if(self.data_main["channels"][ch]["type"] == "brightfield"):
                filename = self.data_main["channels"][ch]["filepaths"][idx]
                return cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return None

    # Load fluorescence images
    def load_images(self, idx):
        output = dict()
        for ch in self.data_main["channels"]:
            filename = self.data_main["channels"][ch]["filepaths"][idx]
            output[self.data_main["channels"][ch]["filenames"][idx]] = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return output

    def filter_droplets(self, circles):
        circles_new = list()
        cirlces_tmp = list()

        droplets_overlapping = list() # These will be marked as overlapping
        droplets_filtered = list() # These will be marked as filtered droplets
        for cidx1 in range(0, len(circles)-1):
            ## This circle has already been filtered by previous operation
            if(cidx1 in droplets_overlapping): 
                continue

            ## Get circle1 properties
            X1 = circles[cidx1][0][0]
            Y1 = circles[cidx1][0][1]
            R1 = circles[cidx1][0][2]

            ## Create a neighbour array to figure out the smallest circle of the candidates
            neighbours = [{"id":cidx1, "R":R1}]
            circle1_overlap = False
            for cidx2 in range(cidx1+1, len(circles)):
                ## This circle has already been filtered by previous operation
                if(cidx2 in droplets_overlapping):
                    continue

                ## Get circle properties
                X2 = circles[cidx2][0][0]
                Y2 = circles[cidx2][0][1]
                R2 = circles[cidx2][0][2]
                D = np.sqrt( (X2-X1)**2 + (Y2-Y1)**2 )

                # Circle overlap check
                if ( D < ( (R1+R2)/2 ) ):
                    circle1_overlap = True
                    droplets_overlapping+=[cidx2]
                    neighbours+=[{"id":cidx2, "R":R2}]
                    # print( "Overlapping: ", cidx1, cidx2, D, R1, R2 )

            # Add circle 1 into the overlapping list
            if circle1_overlap:
                droplets_overlapping += [ cidx1 ]
                sorted_neighbours = sorted(neighbours, key = lambda x : x["R"], reverse=False )
                droplets_filtered += [ sorted_neighbours[0]["id"] ]

        return droplets_overlapping, droplets_filtered

    #Quantify image based on coords and image matrix
    def quantify_image(self, img):
        hmax, wmax = img.shape
        y,x = np.ogrid[-(hmax/2): (hmax/2), -(wmax/2): (wmax/2)]
        mask = x**2+y**2 <= (hmax/2)**2
        mx = ma.masked_array(img, mask=1-mask).compressed()
        return [float(mx.sum()/mx.mean()), float(mx.mean()), float(mx.std()), float(mx.min()), float(mx.max())]

    # Main circle segmentation algorithm
    def get_circles(self, img):
        img_norm = ( 255*( (img - np.min(img))/np.ptp(img)) ).astype(np.uint8)
        # img = cv2.medianBlur(img, 3)

        # Default parameter values: param1=300, param2=0.75
        circles = cv2.HoughCircles(img_norm, cv2.HOUGH_GRADIENT_ALT, 1.5, 10, param1=300, param2=0.75, minRadius=30, maxRadius=80 )
        if circles is None:
            return None
    
        return circles

    # Draws imgs on an image for debugging purposes
    def debug_imgs(self, img, circles, droplets_overlapping, droplets_filtered, name):
        img_norm = ( 255*( (img - np.min(img))/np.ptp(img)) ).astype(np.uint8)
        
        cv2.imwrite( OUTPUT_DIR+"/images/orig_"+name, img_norm )

        cimg = cv2.cvtColor(img_norm,cv2.COLOR_GRAY2BGR)
        circles = np.uint16(np.around(circles))
        for cidx in range(0, len(circles)):
            color_outer = (0,255,0)
            color_inner = (0,0,255)
            if(cidx in droplets_overlapping):
                color_outer = (0,0,255)
                color_inner = (0,255,255)
            if(cidx in droplets_filtered):
                color_outer = (0,255,255)
                color_inner = (0,255,255)
            cv2.circle(cimg, (circles[cidx][0][0], circles[cidx][0][1]), circles[cidx][0][2], color_outer, 1) # draw the outer circle
            cv2.circle(cimg, (circles[cidx][0][0], circles[cidx][0][1]), 2,    color_inner, 1) # draw the center of the circle
        cv2.imwrite( OUTPUT_DIR+"/images/circles_"+name, cimg )


    # Exports raw data in various formats
    def save_results(self):

        if(len(self.results["measurements"])==0):
            print("There are no results to save", file=sys.stderr)
            return None
        
        # List of dicts to dict of lists ...
        # print(self.results["measurements"])
        dict_of_lists = {k: [dic[k] for dic in self.results["measurements"]] for k in self.results["measurements"][0]}
        # Dict of lists to list of dicts: v = [dict(zip(DL,t)) for t in zip(*DL.values())]
        # print(dict_of_lists)
        data_frame = pd.DataFrame.from_dict( dict_of_lists )
        data_frame.to_csv( self.result_dir+"/output.csv" )
        # data_frame.to_json( self.result_dir+"/output.json" )
        
        return True

    # Draws various plots and tables for visualization
    def save_plots(self):
        if(len(self.results["measurements"])==0):
            print("There are no results to save", file=sys.stderr)
            return None
        
        channel_names = [self.data_main["channels"][ch]["name"] for ch in self.data_main["channels"]]

        # List of dicts to dict of lists ...
        dict_of_lists = {k: [dic[k] for dic in self.results["measurements"]] for k in self.results["measurements"][0]}
        data_frame = pd.DataFrame.from_dict( dict_of_lists )

        # Get the column names to export the table
        output_names = {name:name_key for name_key in list(dict_of_lists.keys()) for name in channel_names if name in name_key and "_mean" in name_key}
        output_names["radius_px"] = "radius_px"
        output_names["radius_um"] = "radius_um"
        output_names["image"] = "image"

        data_frame.loc[ :,list(output_names.values()) ].to_json( self.result_dir+"/table_summary.json" )


        # Get the column names to export histograms
        output_names = {name:name_key for name_key in list(dict_of_lists.keys()) for name in channel_names if name in name_key and "_mean" in name_key}
        output_names["radius_px"] = "radius_px"
        output_names["radius_um"] = "radius_um"

        for key in output_names:
            with open(self.result_dir+"/hist_"+key+".json", 'w') as fp:
                hist, bin_edges = np.histogram(data_frame.loc[ :,output_names[key]], density=True, bins=50)
                json.dump({"hist":hist.tolist(), "bins":bin_edges[:-1].tolist()}, fp)

        return True

    ##################################  
    ## Pipeline entry point
    ##################################
    def process(self):
        
        self.results["measurements"] = list()

        crop_id = 0
        image_id = 0
        ## Iterate through the files in the channel file list
        for idx in range(0, self.batch_size):
            print("Processing image: ", idx, "of: ", self.batch_size, file=sys.stderr)
            ## Load brightfield for circle segmentation
            bfg = self.load_brightfield(idx)
            if(bfg is None):
                print("No brightfield image for image nr: ", idx, file=sys.stderr)
                continue
            
            ## Segment brightfield image
            circles = self.get_circles(bfg)
            if(circles is None):
                print("No circles detected for image nr: ", idx, file=sys.stderr)
                continue

            print("Particles detected: ", len(circles), file=sys.stderr)
            droplets_overlapping, droplets_filtered = self.filter_droplets(circles)
            print("Particles detected and filtered: ", len(circles), len(circles)-len(droplets_overlapping)+len(droplets_filtered), file=sys.stderr)
            
            ## Load the image list in preparation to quantify
            image_dict = self.load_images(idx)
            ## Save overlaid images for debugging
            for key in image_dict:
                self.debug_imgs(image_dict[key], circles, droplets_overlapping, droplets_filtered, key.split(".")[0]+".png")

            print("Images loaded: ", len(image_dict), file=sys.stderr)
            ## iterate through the particles and loaded images
            
            for cidx in range(0, len(circles)):
                tmp_data = dict()
                tmp_data["image"] = image_id
                tmp_data["crop_id"] = crop_id
                tmp_data["particle_id"] = cidx
                tmp_data["X"] = int(circles[cidx][0][0])
                tmp_data["Y"] = int(circles[cidx][0][1])
                tmp_data["radius_px"] = int(circles[cidx][0][2])
                tmp_data["radius_um"] = float(tmp_data["radius_px"]/self.scale)
                tmp_data["overlapping"] = cidx in droplets_overlapping
                tmp_data["filtered"] = (not tmp_data["overlapping"]) or (cidx in droplets_filtered)
                
                ha = int((tmp_data["Y"]-tmp_data["radius_px"]))
                hb = int((tmp_data["Y"]+tmp_data["radius_px"]))
                ja = int((tmp_data["X"]-tmp_data["radius_px"]))
                jb = int((tmp_data["X"]+tmp_data["radius_px"]))
                hmax, wmax = bfg.shape
                if(ha < 0):
                    continue
                if(hb > hmax):
                    continue
                if(ja < 0):
                    continue
                if(jb > wmax):
                    continue

                ## Measure particle quantities from the loaded images
                crop_img = np.zeros( (CROP_SIZE, CROP_SIZE*len(self.data_main["channels"]) ), dtype = np.uint16)
                crop_orig_img = np.zeros( (CROP_SIZE, CROP_SIZE*len(self.data_main["channels"]) ), dtype = np.uint16)
                crop_idx = 0
                filename = ""
                for ch in self.data_main["channels"]:
                    ch_name =self.data_main["channels"][ch]["name"]
                    name_ch = self.data_main["channels"][ch]["filenames"][idx]
                    tmp_data[ch_name + "_px"], tmp_data[ch_name + "_mean"], tmp_data[ch_name + "_sd"], tmp_data[ch_name + "_min"], tmp_data[ch_name + "_max"] = self.quantify_image( image_dict[name_ch][ha:hb, ja:jb])
                    
                    if(tmp_data["filtered"]):
                        img_tmp = cv2.resize( image_dict[name_ch][ha:hb, ja:jb], (CROP_SIZE, CROP_SIZE), interpolation = cv2.INTER_AREA ).astype(np.uint16)
                        crop_img[0:CROP_SIZE, (crop_idx*CROP_SIZE):((crop_idx+1)*CROP_SIZE)] = ( 65535*( (img_tmp - np.min(img_tmp))/np.ptp(img_tmp)) ).astype(np.uint16) 
                        crop_orig_img[0:CROP_SIZE, (crop_idx*CROP_SIZE):((crop_idx+1)*CROP_SIZE)] = img_tmp.astype(np.uint16)
                        
                        crop_idx+=1
                        if crop_idx == 1:
                            filename = ch
                        else:
                            filename = filename + "_" + ch

                if tmp_data["filtered"] and os.path.exists( os.path.join(self.result_dir, "crops") ) and os.path.exists( os.path.join(self.result_dir, "crops_orig") ):
                    cv2.imwrite( os.path.join(self.result_dir, "crops",  filename+"-"+str(crop_id)+"-"+str(image_id)+".png"), crop_img)
                    cv2.imwrite( os.path.join(self.result_dir, "crops_orig",  filename+"-"+str(crop_id)+"-"+str(image_id)+".png"), crop_orig_img)

                crop_id+=1

                self.results["measurements"].append(tmp_data)
            #     break
            # break
            image_id+=1
        
        # with open(os.path.join(sys.path[0], self.result_dir, "results.json"), 'w') as fp:
        #     json.dump(self.results, fp)

        return True




if __name__ == '__main__':
    
    FRAMEWORK = Framework()
    FRAMEWORK.process()
    FRAMEWORK.save_results()
    # FRAMEWORK.save_plots()