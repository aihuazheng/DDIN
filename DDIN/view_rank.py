class Visualization:

    def __init__(self, query_img_, gall_img_, distmat, q_pids, g_pids):
        self.query_img = query_img
        self.gall_img = gall_img
        self.distmat = distmat
        
        self.indices_ = np.argsort(distmat, axis=1)
        self.matches_ = (g_pids[self.indices_] == q_pids[:, np.newaxis]).astype(np.int32)
        
    # from the start-th to end-th row
    def save_figs(self, start, end):
        
        gall_img = np.asarray(self.gall_img)
        query_img = np.asarray(self.query_img)
        
        gallery = gall_img[self.indices_][:, :10]
        
        imgs_path = np.hstack([query_img.reshape(-1, 1), gallery])
        
        for row in range(start-1, end+1):
            save_single_fig(row, imgs_path, self.matches)
    
        # for one row
    def save_single_fig(self, which_row, imgs_path, self.matches):
        
        fig, ax = plt.subplots(1, 11, figsize = (40,10))
        for i, axi in enumerate(ax.flat):
            
            # no border color
            if i == 0:
                img = read_img(imgs_path[which_row, 0], "none")
            
            # border color: 0 == "red" 1 == "green"
            else:
                if self.indices_[which_row, i-1] == 0:
                    img = read_img(imgs_path[which_row, i], "red")
                else:
                    img = read_img(imgs_path[which_row, i], "green")
            
            #pdb.set_trace()
            w, h, c = img.shape[1], img.shape[2], img.shape[3]
            axi.imshow(img.reshape(w, h, c))
            axi.set(xticks=[], yticks=[])
        
        fig.savefig("result-" + str(which_row) + ".jpg")
    
    def read_img(self, img_path, color):
    
        img = Image.open(img_path)
        img = img.resize((144, 288))
        
        if color == "red":
          img = ImageOps.expand(img, border=15, fill='red')##left,top,right,bottom
        elif color == "green":
          img = ImageOps.expand(img, border=15, fill='green')##left,top,right,bottom
        else:
          img = I