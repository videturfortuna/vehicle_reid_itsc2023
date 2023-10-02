from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QComboBox, QLabel, QSpinBox, QLCDNumber, QSlider
import numpy as np
from glob import glob
import os


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 900)

        self.centralwidget = QtWidgets.QWidget(MainWindow)

        self.combDataset= QtWidgets.QComboBox(self.centralwidget)
        self.combDataset.setGeometry(QtCore.QRect(620, 10, 151, 51))
        self.combDataset.addItem("Veri776")
        self.combDataset.addItem("VERIWILD")
        self.combDataset.activated.connect(self.update_options)
        #self.combDataset.activated.connect(self.show_query)

        self.path_w = "./logs/Veri776/4G/0/"
        
        with open(self.path_w +'result_map_l2_'+ str(True) + '_mean_' + str(False) +'.npy', 'rb') as f:
            self.mAP = np.load(f)
        with open(self.path_w +'result_cmc_l2_'+ str(True) + '_mean_' + str(False) +'.npy', 'rb') as f:
            self.cmc = np.load(f)

        self.query_dir = "/home/eurico/VeRi/image_query/"
        self.gallery_dir = "/home/eurico/VeRi/image_test/"


        with open(self.path_w + 'distmat.npy', 'rb') as f:
            self.distmat = np.load(f)
        # with open(self.path_w + 'q_view.npy', 'rb') as f:
        #     self.q_view = np.load(f)
        # with open(self.path_w + 'g_view.npy', 'rb') as f:
        #     self.g_view = np.load(f)
        self.q_view = np.zeros(self.distmat.shape[0])
        self.g_view = np.zeros(self.distmat.shape[1])
        self.min_dist_idx = np.argsort(self.distmat, axis = 1)

        self.vw_q_imgpath = np.loadtxt('./train_test_split/test_3000_id_query.txt', dtype='str_')
        self.vw_g_imgpath = np.loadtxt('./train_test_split/test_3000_id.txt', dtype='str_')
        self.v776_q = np.loadtxt('/home/eurico/VeRi/name_query.txt', dtype='str_')
        self.v776_g = np.loadtxt('/home/eurico/VeRi/name_test.txt', dtype='str_')


        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(520, 30, 76, 19))


        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(210, 270, 61, 21))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.font_bold = font
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(False)
        self.font = font

        self.label_2.setFont(font)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(930, 70, 111, 19))

        self.label_3.setFont(font)
        self.img_query = QtWidgets.QLabel(self.centralwidget)
        self.img_query.setGeometry(QtCore.QRect(110, 330, 256, 256))
        self.img_query.setText("")
        #self.img_query.setPixmap(QtGui.QPixmap(self.path_w + "activations/0.jpg"))
        self.img_query.setObjectName("img_query")
        self.img_query.setScaledContents(True)


        self.querylabel_info = QtWidgets.QLabel(self.centralwidget)
        self.querylabel_info.setGeometry(QtCore.QRect(75, 300, 400, 30))

        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox.setGeometry(QtCore.QRect(280, 270, 56, 30))
        self.spinBox.setMaximum(self.distmat.shape[0]-1)
        self.spinBox.valueChanged.connect(self.show_query)

        self.img_g1 = QtWidgets.QLabel(self.centralwidget)
        self.img_g1.setGeometry(QtCore.QRect(600, 140, 256, 256))
        self.img_g1.setScaledContents(True)
        self.img_g1_info = QtWidgets.QLabel(self.centralwidget)
        self.img_g1_info.setGeometry(QtCore.QRect(600, 420, 200, 30))


        self.img_g2 = QtWidgets.QLabel(self.centralwidget)
        self.img_g2.setGeometry(QtCore.QRect(890, 140, 256, 256))
        self.img_g2.setScaledContents(True)
        self.img_g2_info = QtWidgets.QLabel(self.centralwidget)
        self.img_g2_info.setGeometry(QtCore.QRect(890, 420, 200, 30))


        self.img_g3 = QtWidgets.QLabel(self.centralwidget)
        self.img_g3.setGeometry(QtCore.QRect(1170, 140, 256, 256))
        self.img_g3.setScaledContents(True)
        self.img_g3_info = QtWidgets.QLabel(self.centralwidget)
        self.img_g3_info.setGeometry(QtCore.QRect(1170, 420, 200, 30))


        self.img_g4 = QtWidgets.QLabel(self.centralwidget)
        self.img_g4.setGeometry(QtCore.QRect(600, 500, 256, 256))
        self.img_g4.setScaledContents(True)
        self.img_g4_info = QtWidgets.QLabel(self.centralwidget)
        self.img_g4_info.setGeometry(QtCore.QRect(600, 780, 200, 30))
  

        self.img_g5 = QtWidgets.QLabel(self.centralwidget)
        self.img_g5.setGeometry(QtCore.QRect(880,500,256,256))
        self.img_g5.setScaledContents(True)
        self.img_g5_info = QtWidgets.QLabel(self.centralwidget)
        self.img_g5_info.setGeometry(QtCore.QRect(880, 780, 200, 30))


        self.img_g6 = QtWidgets.QLabel(self.centralwidget)
        self.img_g6.setGeometry(QtCore.QRect(1170, 500, 256, 256))
        self.img_g6.setScaledContents(True)
        self.img_g6_info = QtWidgets.QLabel(self.centralwidget)
        self.img_g6_info.setGeometry(QtCore.QRect(1170, 780, 200, 30))





        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(600, 460, 91, 20))
        self.label_12.setScaledContents(True)
        self.lcdNumber_2 = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber_2.setGeometry(QtCore.QRect(690, 460, 64, 23))


        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(880, 460, 91, 20))
        self.lcdNumber_3 = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber_3.setGeometry(QtCore.QRect(970, 460, 64, 23))


        self.lcdNumber_4 = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber_4.setGeometry(QtCore.QRect(1260, 460, 64, 23))
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(1170, 460, 91, 20))
        

        self.lcdNumber_5 = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber_5.setGeometry(QtCore.QRect(690, 820, 64, 23))
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(600, 820, 91, 20))

        self.lcdNumber_6 = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber_6.setGeometry(QtCore.QRect(970, 820, 64, 23))
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(880, 820, 91, 20))

        self.lcdNumber_7 = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber_7.setGeometry(QtCore.QRect(1260, 820, 64, 23))
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(1170, 820, 91, 20))


        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setGeometry(QtCore.QRect(160, 620, 151, 51))
        self.comboBox_2.addItem("Original")
        self.comboBox_2.addItem("all_branch")
        self.comboBox_2.addItem("ce")
        self.comboBox_2.addItem("triplet")
        self.comboBox_2.addItem("transform_ce_triplet")
        self.comboBox_2.addItem("transform_triplet")
        self.comboBox_2.activated.connect(self.show_query)

        self.comboBox_3 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_3.setGeometry(QtCore.QRect(1210, 10, 151, 51))

        self.comboBox_4 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_4.setGeometry(QtCore.QRect(150, 10, 361, 51))

        self.comboBox_4.addItem("4G - Hybrid")
        self.comboBox_4.addItem("2G - R50")
        self.comboBox_4.addItem("1B - Baseline R50")
        self.comboBox_4.addItem("2x2G - R50 LSB")
        self.comboBox_4.addItem("2B - Baselines R50")
        self.comboBox_4.addItem("2B - R50")
        self.comboBox_4.addItem("4B")
        self.comboBox_4.addItem("4B_alllosses")

        self.comboBox_4.activated.connect(self.update_options)

        self.list_arch = ['MBR_4G', 'MBR_R50_2G', 'Baseline', 'MBR_R50_2x2G', 'R50_2B', 'MBR_R50_2B', 'MBR_4B', 'Hybrid_4B']
        for paths in glob('./logs/'+self.combDataset.currentText()+'/'+self.list_arch[self.comboBox_4.currentIndex()]+'/*/distmat.npy'):
            self.comboBox_3.addItem(paths[:-11])
        self.comboBox_3.activated.connect(self.changeDataset)
        self.comboBox_3.activated.connect(self.show_query)

        self.sl = QtWidgets.QSlider(self.centralwidget)
        self.sl.setGeometry(QtCore.QRect(1150, 70, 221, 16))
        self.sl.setOrientation(QtCore.Qt.Horizontal)
        self.sl.setMinimum(0)
        self.sl.setMaximum(100)
        self.sl.setValue(0)
        self.sl.setSingleStep(6)
        #self.sl.setTickPosition(QtWidgets.QSlider.TicksBelow)
        #self.sl.setTickInterval(6)
        self.sl.valueChanged.connect(self.show_query)





        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(1110, 30, 76, 19))

        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(40, 20, 101, 19))


        

        self.lcdNumber_8 = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber_8.setGeometry(QtCore.QRect(300, 140, 64, 23))
        self.lcdNumber_8.display('{:.02f}'.format(self.cmc[4]*100))
        self.lcdNumber_8.setStyleSheet("""QLCDNumber 
                                        { background-color: black; 
                                            color: white;
                                        }""")  
        self.lcdNumber_9 = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber_9.setGeometry(QtCore.QRect(230, 140, 64, 23))
        self.lcdNumber_9.display('{:.02f}'.format(self.cmc[0]*100))
        self.lcdNumber_9.setStyleSheet("""QLCDNumber 
                                        { background-color: black; 
                                            color: white;
                                        }""")  
        self.lcdNumber_10 = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcdNumber_10.setGeometry(QtCore.QRect(160, 140, 64, 23))
        self.lcdNumber_10.display('{:.02f}'.format(self.mAP*100))
        self.lcdNumber_10.setStyleSheet("""QLCDNumber 
                                        { background-color: black; 
                                            color: white;
                                        }""")        

        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(60, 140, 76, 19))

        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(170, 110, 41, 19))

        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(240, 110, 51, 19))

        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(310, 110, 51, 19))

        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(140, 680, 181, 25))
        
        self.checkBox.stateChanged.connect(self.show_query)
        

        self.checkBoxFailedcmc1 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBoxFailedcmc1.setGeometry(QtCore.QRect(140, 730, 181, 25))
        self.checkBoxFailedcmc1.stateChanged.connect(self.show_query)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1441, 24))

        self.menuReID = QtWidgets.QMenu(self.menubar)

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)

        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuReID.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.combDataset.setItemText(0, _translate("MainWindow", "Veri776"))
        self.combDataset.setItemText(1, _translate("MainWindow", "VERIWILD"))
        self.label.setText(_translate("MainWindow", "Dataset"))
        self.label_2.setText(_translate("MainWindow", "Query"))
        self.label_3.setText(_translate("MainWindow", "Ranking List"))
        self.img_g1.setText(_translate("MainWindow", "Gallery image"))
        self.img_g2.setText(_translate("MainWindow", "Gallery image"))
        self.img_g3.setText(_translate("MainWindow", "Gallery image"))
        self.img_g4.setText(_translate("MainWindow", "Gallery image"))
        self.img_g5.setText(_translate("MainWindow", "Gallery image"))
        self.img_g6.setText(_translate("MainWindow", "Gallery image"))
        self.label_12.setText(_translate("MainWindow", "Similarity"))
        self.label_13.setText(_translate("MainWindow", "Similarity"))
        self.label_14.setText(_translate("MainWindow", "Similarity"))
        self.label_15.setText(_translate("MainWindow", "Similarity"))
        self.label_16.setText(_translate("MainWindow", "Similarity"))
        self.label_17.setText(_translate("MainWindow", "Similarity"))
        # self.comboBox_2.setItemText(0, _translate("MainWindow", "Original"))
        # self.comboBox_2.setItemText(1, _translate("MainWindow", "all_branch"))
        # self.comboBox_2.setItemText(2, _translate("MainWindow", "act_t_Brand"))
        # self.comboBox_2.setItemText(3, _translate("MainWindow", "ce_branch"))

        self.label_5.setText(_translate("MainWindow", "Weights"))
        self.label_19.setText(_translate("MainWindow", "Score"))
        self.label_20.setText(_translate("MainWindow", "mAP"))
        self.label_21.setText(_translate("MainWindow", "CMC1"))
        self.label_22.setText(_translate("MainWindow", "CMC5"))
        self.checkBox.setText(_translate("MainWindow", "Exclude same view"))
        self.checkBoxFailedcmc1.setText(_translate("MainWindow", "CMC1 Fails only"))
        self.menuReID.setTitle(_translate("MainWindow", "ReID"))

    def get_activation_dir(self, idx):
        
        if self.comboBox_2.currentText() == 'Original':
            if self.combDataset.currentText() == "VERIWILD":
                path2img_g = self.gallery_dir+ self.vw_g_imgpath[idx, 0]
            else:
                path2img_g = self.gallery_dir+ str(self.v776_g[idx])
        elif self.comboBox_2.currentText() == 'all_branch':
            path2img_g = self.path_w + "activations_g/"+ str(idx)+ ".jpg"
        elif self.comboBox_2.currentText() == 'ce':
            path2img_g = self.path_w + "activations_g/cross_entropy_branch/"+ str(idx)+ ".jpg"
        elif self.comboBox_2.currentText() == 'triplet':
            path2img_g = self.path_w + "activations_g/triplet_branch/"+ str(idx)+ ".jpg"
        elif self.comboBox_2.currentText() == 'transform_ce_triplet':
            if os.path.exists(self.path_w + "activations_g/mhsa_branch/"+ str(idx)+ ".jpg"):
                path2img_g = self.path_w + "activations_g/mhsa_branch/"+ str(idx)+ ".jpg"
            else:
                path2img_g = self.path_w + "activations_g/mhsa_ce_branch/"+ str(idx)+ ".jpg"   
        elif self.comboBox_2.currentText() == 'transform_triplet':
            path2img_g = self.path_w + "activations_g/mhsa_t_branch/"+ str(idx)+ ".jpg"
     
        return path2img_g

    def set_galleryimg(self, g_idx, lcd, g_image, g_image_info, q_ID, q_idx, q_pic):
        if self.combDataset.currentText() == "VERIWILD":
            path2img_g = self.get_activation_dir(g_idx)
            g_id = self.vw_g_imgpath[g_idx, 1]
            view_g = self.vw_g_imgpath[g_idx, 3]
            gallery_cam = self.vw_g_imgpath[g_idx, 2]
            self.img_query.setPixmap(q_pic)
            g_image.setPixmap(QtGui.QPixmap(path2img_g))
            if g_id == q_ID:
                lcd.display('{:.02f}'.format(self.distmat[q_idx, g_idx]))
                lcd.setStyleSheet("""QLCDNumber 
                                                { background-color: green; 
                                                    color: yellow;
                                                }""")
            else:
                lcd.display('{:.02f}'.format(self.distmat[q_idx, g_idx]))
                lcd.setStyleSheet("""QLCDNumber 
                                                { background-color: red; 
                                                    color: yellow;
                                                }""")
        else:
            path2img_g = self.get_activation_dir(g_idx)
            g_id = str(self.v776_g[g_idx])[:4]
            view_g = self.g_view[g_idx]
            gallery_cam = str(self.v776_g[g_idx])[5:9]
            g_image.setPixmap(QtGui.QPixmap(path2img_g))
            if g_id == q_ID:
                lcd.display('{:.02f}'.format(self.distmat[q_idx, g_idx]))
                lcd.setStyleSheet("""QLCDNumber 
                                                { background-color: green; 
                                                    color: yellow;
                                                }""")
            else:
                lcd.display('{:.02f}'.format(self.distmat[q_idx, g_idx]))
                lcd.setStyleSheet("""QLCDNumber 
                                                { background-color: red; 
                                                    color: yellow;
                                                }""")
        g_image_info.setText("ID: "+ str(g_id)+ "  View: "+ str(view_g) + "  Cam: "+ str(gallery_cam))
    

    def show_query(self):
        
        q_idx = self.spinBox.value()
        
        if self.combDataset.currentText() == "VERIWILD":
            ranklist = self.min_dist_idx[q_idx, :1000]
            q_names = self.vw_q_imgpath[:,0]
            q_ID = self.vw_q_imgpath[q_idx,1]
            query_cam = self.vw_q_imgpath[q_idx,2]
            view_q = self.vw_q_imgpath[q_idx,3]
            g_pids = np.int32(self.vw_g_imgpath[:,1])
            g_camids =  np.int32(self.vw_g_imgpath[:,2])
            g_viewids = np.int32(self.vw_g_imgpath[:,3])
            # if self.checkBoxFailedcmc1.isChecked():
            #     while int(q_ID) == g_pids[ranklist[0]]:
            #         q_idx +=1
            #         q_ID = str(q_names[q_idx])[:4]
            #         query_cam = str(q_names[q_idx])[5:9]
            #         view_q = self.q_view[q_idx]
            #         ranklist = self.min_dist_idx[q_idx, :1000]
        else:
            
            ranklist = self.min_dist_idx[q_idx, :1000]
            q_names = self.v776_q
            q_ID = str(q_names[q_idx])[:4]
            query_cam = str(q_names[q_idx])[5:9]
            view_q = self.q_view[q_idx]
            g_pids = np.int32(np.asarray([pid[:4] for pid in self.v776_g]))
            g_camids =  np.int32(np.asarray([pid[6:9] for pid in self.v776_g]))
            g_viewids =  np.int32(self.g_view)
            # if self.checkBoxFailedcmc1.isChecked():
            #     cnt = 0
            #     while query_cam==g_camids[ranklist[cnt]]:
            #         cnt += 1
            #     while int(q_ID) == g_pids[ranklist[cnt]]:
            #         q_idx +=1
            #         q_ID = str(q_names[q_idx])[:4]
            #         query_cam = str(q_names[q_idx])[5:9]
            #         view_q = self.q_view[q_idx]
            #         ranklist = self.min_dist_idx[q_idx, :1000]
                    

        if self.comboBox_2.currentText() == 'all_branch':
            pic = QtGui.QPixmap(self.path_w + "activations/"+ str(q_idx)+ ".jpg")
            self.img_query.setPixmap(pic)
        elif self.comboBox_2.currentText() == 'Original':
            pic = QtGui.QPixmap(self.query_dir + str(q_names[q_idx]))
            self.img_query.setPixmap(pic)
        elif self.comboBox_2.currentText() == 'ce':
            pic = QtGui.QPixmap(self.path_w + "activations/cross_entropy_branch/"+ str(q_idx)+ ".jpg")
            self.img_query.setPixmap(pic)
        elif self.comboBox_2.currentText() == 'triplet':
            pic = QtGui.QPixmap(self.path_w + "activations/triplet_branch/"+ str(q_idx)+ ".jpg")
            self.img_query.setPixmap(pic)      
        elif self.comboBox_2.currentText() == 'transform_ce_triplet':
            if os.path.exists(self.path_w + "activations/mhsa_branch/"+ str(q_idx)+ ".jpg"):
                pic = QtGui.QPixmap(self.path_w + "activations/mhsa_branch/"+ str(q_idx)+ ".jpg")
            else:
                pic = QtGui.QPixmap(self.path_w + "activations/mhsa_ce_branch/"+ str(q_idx)+ ".jpg")
            self.img_query.setPixmap(pic)   
        elif self.comboBox_2.currentText() == 'transform_triplet':
            pic = QtGui.QPixmap(self.path_w + "activations/mhsa_t_branch/"+ str(q_idx)+ ".jpg")
            self.img_query.setPixmap(pic)       

        new_ranklist = []
        ##  nimages to match
        count_nquery_in_g = 0
        if self.combDataset.currentText() == "VERIWILD":
            for conta, i in enumerate(self.vw_g_imgpath[:, 1]):
                if q_ID == i:
                    if not (g_camids[conta] == np.int32(query_cam)):
                        if self.checkBox.isChecked():
                            if not (g_viewids[conta] == np.int32(view_q)):
                                count_nquery_in_g += 1 
                        else:
                            count_nquery_in_g += 1   
        else:
            for conta, i in enumerate(self.v776_g):
                if q_ID in i[:4]:
                    if not (g_camids[conta] == np.int32(query_cam[1:])):
                        if self.checkBox.isChecked():
                            if not (g_viewids[conta] == view_q):
                                count_nquery_in_g += 1 
                        else:
                            count_nquery_in_g += 1

        # print(str(count_nquery_in_g), ' images of ID ', q_ID, ' present in gallery')

        ## compute AP for sample q_ID
        order = np.argsort(self.distmat[q_idx])
        if self.combDataset.currentText() == "VERIWILD":
            if self.checkBox.isChecked():
                remove = (g_pids[order] == np.int32(q_ID)) & (g_camids[order] == np.int32(query_cam) & (g_viewids[order] == np.int32(view_q)))
            else:
                remove = (g_pids[order] == np.int32(q_ID)) & (g_camids[order] == np.int32(query_cam))
        else:
            if self.checkBox.isChecked():
                remove = (g_pids[order] == np.int32(q_ID)) & (g_camids[order] == np.int32(query_cam[1:]) & (g_viewids[order] == view_q))
            else:
                remove = (g_pids[order] == np.int32(q_ID)) & (g_camids[order] == np.int32(query_cam[1:]))
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = (g_pids[order] == np.int32(q_ID)).astype(np.int32)[keep]

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        # compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel

        self.querylabel_info.setText("ID: "+ str(q_ID)+ "  View: "+ str(view_q) + "  Cam: "+ str(query_cam) + "  Q_AP: " + f'{AP*100:3.1f}' + " Nimg in gal: " + str(count_nquery_in_g))


        for count, g_idx in enumerate(ranklist):
            if self.combDataset.currentText() == "VERIWILD":
                g_id = self.vw_g_imgpath[g_idx, 1]
                view_g = self.vw_g_imgpath[g_idx, 3]
                gallery_cam = self.vw_g_imgpath[g_idx, 2]
                if self.checkBox.isChecked():
                    if (gallery_cam == query_cam) or (view_q == view_g):
                        continue
                    new_ranklist.append(g_idx)
                else:
                    if (gallery_cam == query_cam):
                        continue
                    new_ranklist.append(g_idx)
            else:
                g_id = str(self.v776_g[g_idx])[:4]
                view_g = self.g_view[g_idx]
                gallery_cam = str(self.v776_g[g_idx])[5:9]
                if self.checkBox.isChecked():
                    if (gallery_cam == query_cam) or (view_q == view_g):
                        continue
                    new_ranklist.append(g_idx)
                else:
                    if (gallery_cam == query_cam):
                        continue
                    new_ranklist.append(g_idx)

        start = self.sl.value()
        g_idx=new_ranklist[start]
        self.set_galleryimg(g_idx, self.lcdNumber_2, self.img_g1, self.img_g1_info, q_ID, q_idx, pic)
        
        g_idx=new_ranklist[start+1]
        self.set_galleryimg(g_idx, self.lcdNumber_3, self.img_g2, self.img_g2_info, q_ID, q_idx, pic)

        g_idx=new_ranklist[start+2]
        self.set_galleryimg(g_idx, self.lcdNumber_4, self.img_g3, self.img_g3_info, q_ID, q_idx, pic)

        g_idx=new_ranklist[start+3]
        self.set_galleryimg(g_idx, self.lcdNumber_5, self.img_g4, self.img_g4_info, q_ID, q_idx, pic)

        g_idx=new_ranklist[start+4]
        self.set_galleryimg(g_idx, self.lcdNumber_6, self.img_g5, self.img_g5_info, q_ID, q_idx, pic)

        g_idx=new_ranklist[start+5]
        self.set_galleryimg(g_idx, self.lcdNumber_7, self.img_g6, self.img_g6_info, q_ID, q_idx, pic)

    def show_ranking(self):
        print('debug')

    def update_options(self):
        self.comboBox_3.clear()
        for paths in glob('./logs/'+self.combDataset.currentText()+'/'+self.list_arch[self.comboBox_4.currentIndex()]+'/*/distmat.npy'):
            self.comboBox_3.addItem(paths[:-11])
    def changeDataset(self):
        if self.combDataset.currentText() == 'VERIWILD':
            self.path_w = self.comboBox_3.currentText()
            self.query_dir = "/home/eurico/VERI-Wild/images/"
            self.gallery_dir = "/home/eurico//VERI-Wild/images/"
            if self.comboBox_4.	currentText() == "1B - Baseline":
                with open(self.path_w +'result_map_l2_'+ str(True) + '_mean_' + str(False) +'.npy', 'rb') as f:
                    self.mAP = np.load(f)
                with open(self.path_w +'result_cmc_l2_'+ str(True) + '_mean_' + str(False) +'.npy', 'rb') as f:
                    self.cmc = np.load(f)    
            else:
                with open(self.path_w +'result_map_l2_'+ str(True) + '_mean_' + str(False) +'.npy', 'rb') as f:
                    self.mAP = np.load(f)
                with open(self.path_w +'result_cmc_l2_'+ str(True) + '_mean_' + str(False) +'.npy', 'rb') as f:
                    self.cmc = np.load(f)    
    
            with open(self.path_w + 'distmat.npy', 'rb') as f:
                self.distmat = np.load(f)
            # with open(self.path_w + 'q_view.npy', 'rb') as f:
            #     self.q_view = np.load(f)
            # with open(self.path_w + 'g_view.npy', 'rb') as f:
            #     self.g_view = np.load(f)
            self.q_view = np.zeros(self.distmat.shape[0])
            self.g_view = np.zeros(self.distmat.shape[1])
            self.lcdNumber_10.display('{:.02f}'.format(self.mAP*100))
            self.lcdNumber_9.display('{:.02f}'.format(self.cmc[0]*100))
            self.lcdNumber_8.display('{:.02f}'.format(self.cmc[4]*100))    
            self.min_dist_idx = np.argsort(self.distmat, axis = 1)
        else:            
            self.path_w = self.comboBox_3.currentText()
            self.query_dir = "/home/eurico/VeRi/image_query/"
            self.gallery_dir = "/home/eurico/VeRi/image_test/"

            with open(self.path_w + 'distmat.npy', 'rb') as f:
                self.distmat = np.load(f)
            # with open(self.path_w + 'q_view.npy', 'rb') as f:
            #     self.q_view = np.load(f)
            # with open(self.path_w + 'g_view.npy', 'rb') as f:
            #     self.g_view = np.load(f)
            self.q_view = np.zeros(self.distmat.shape[0])
            self.g_view = np.zeros(self.distmat.shape[1])
            with open(self.path_w +'result_map_l2_'+ str(True) + '_mean_' + str(False) +'.npy', 'rb') as f:
                self.mAP = np.load(f)
            with open(self.path_w +'result_cmc_l2_'+ str(True) + '_mean_' + str(False) +'.npy', 'rb') as f:
                self.cmc = np.load(f)
            self.lcdNumber_10.display('{:.02f}'.format(self.mAP*100))
            self.lcdNumber_9.display('{:.02f}'.format(self.cmc[0]*100))
            self.lcdNumber_8.display('{:.02f}'.format(self.cmc[4]*100))
            self.min_dist_idx = np.argsort(self.distmat, axis = 1)




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

