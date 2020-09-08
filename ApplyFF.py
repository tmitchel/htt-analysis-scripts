import ROOT

class FFApplicationTool():
    def __init__(self,theFFDirectory,channel,isDifferential=False,attempt0JetMETCorrection=False):
        self.theFFDirectory = theFFDirectory
        self.isDifferential = isDifferential
        self.attempt0JetMETCorrection = attempt0JetMETCorrection
        
        self.theRawFile = ROOT.TFile(theFFDirectory+"uncorrected_fakefactors_"+channel+".root")
        if self.theRawFile.IsZombie():            
            raise RuntimeError("Problem loading the files!")
        self.ff_qcd_0jet = self.theRawFile.Get("rawFF_"+channel+"_qcd_0jet")
        self.ff_qcd_0jet_unc1_up = self.theRawFile.Get("rawFF_"+channel+"_qcd_0jet_unc1_up")
        self.ff_qcd_0jet_unc1_down = self.theRawFile.Get("rawFF_"+channel+"_qcd_0jet_unc1_down")
        self.ff_qcd_0jet_unc2_up = self.theRawFile.Get("rawFF_"+channel+"_qcd_0jet_unc2_up")
        self.ff_qcd_0jet_unc2_down = self.theRawFile.Get("rawFF_"+channel+"_qcd_0jet_unc2_down")

        self.ff_qcd_1jet = self.theRawFile.Get("rawFF_"+channel+"_qcd_1jet")
        self.ff_qcd_1jet_unc1_up = self.theRawFile.Get("rawFF_"+channel+"_qcd_1jet_unc1_up")
        self.ff_qcd_1jet_unc1_down = self.theRawFile.Get("rawFF_"+channel+"_qcd_1jet_unc1_down")
        self.ff_qcd_1jet_unc2_up = self.theRawFile.Get("rawFF_"+channel+"_qcd_1jet_unc2_up")
        self.ff_qcd_1jet_unc2_down = self.theRawFile.Get("rawFF_"+channel+"_qcd_1jet_unc2_down")

        self.ff_qcd_2jet = self.theRawFile.Get("rawFF_"+channel+"_qcd_2jet")
        self.ff_qcd_2jet_unc1_up = self.theRawFile.Get("rawFF_"+channel+"_qcd_2jet_unc1_up")
        self.ff_qcd_2jet_unc1_down = self.theRawFile.Get("rawFF_"+channel+"_qcd_2jet_unc1_down")
        self.ff_qcd_2jet_unc2_up = self.theRawFile.Get("rawFF_"+channel+"_qcd_2jet_unc2_up")
        self.ff_qcd_2jet_unc2_down = self.theRawFile.Get("rawFF_"+channel+"_qcd_2jet_unc2_down")

        self.ff_w_0jet = self.theRawFile.Get("rawFF_"+channel+"_w_0jet")
        self.ff_w_0jet_unc1_up = self.theRawFile.Get("rawFF_"+channel+"_w_0jet_unc1_up")
        self.ff_w_0jet_unc1_down = self.theRawFile.Get("rawFF_"+channel+"_w_0jet_unc1_down")
        self.ff_w_0jet_unc2_up = self.theRawFile.Get("rawFF_"+channel+"_w_0jet_unc2_up")
        self.ff_w_0jet_unc2_down = self.theRawFile.Get("rawFF_"+channel+"_w_0jet_unc2_down")

        self.ff_w_1jet = self.theRawFile.Get("rawFF_"+channel+"_w_1jet")
        self.ff_w_1jet_unc1_up = self.theRawFile.Get("rawFF_"+channel+"_w_1jet_unc1_up")
        self.ff_w_1jet_unc1_down = self.theRawFile.Get("rawFF_"+channel+"_w_1jet_unc1_down")
        self.ff_w_1jet_unc2_up = self.theRawFile.Get("rawFF_"+channel+"_w_1jet_unc2_up")
        self.ff_w_1jet_unc2_down = self.theRawFile.Get("rawFF_"+channel+"_w_1jet_unc2_down")

        self.ff_w_2jet = self.theRawFile.Get("rawFF_"+channel+"_w_2jet")
        self.ff_w_2jet_unc1_up = self.theRawFile.Get("rawFF_"+channel+"_w_2jet_unc1_up")
        self.ff_w_2jet_unc1_down = self.theRawFile.Get("rawFF_"+channel+"_w_2jet_unc1_down")
        self.ff_w_2jet_unc2_up = self.theRawFile.Get("rawFF_"+channel+"_w_2jet_unc2_up")
        self.ff_w_2jet_unc2_down = self.theRawFile.Get("rawFF_"+channel+"_w_2jet_unc2_down")

        self.ff_tt_0jet = self.theRawFile.Get("mc_rawFF_"+channel+"_tt")
        self.ff_tt_0jet_unc1_up = self.theRawFile.Get("mc_rawFF_"+channel+"_tt_unc1_up")
        self.ff_tt_0jet_unc1_down = self.theRawFile.Get("mc_rawFF_"+channel+"_tt_unc1_down")
        self.ff_tt_0jet_unc2_up = self.theRawFile.Get("mc_rawFF_"+channel+"_tt_unc2_up")
        self.ff_tt_0jet_unc2_down = self.theRawFile.Get("mc_rawFF_"+channel+"_tt_unc2_down")
        
        #FIXME
        self.theFMvisFile = ROOT.TFile(theFFDirectory+"FF_corrections_1.root")
        if self.theFMvisFile.IsZombie():
            raise RuntimeError("Problem loading the files!")
        
        #MVis closure
        self.mVisClosure_QCD_0jet = self.theFMvisFile.Get("closure_mvis_"+channel+"_0jet_qcd")
        self.mVisClosure_QCD_1jet = self.theFMvisFile.Get("closure_mvis_"+channel+"_1jet_qcd")
        self.mVisClosure_QCD_2jet = self.theFMvisFile.Get("closure_mvis_"+channel+"_2jet_qcd")
        self.mVisClosure_W_0jet = self.theFMvisFile.Get("closure_mvis_"+channel+"_0jet_w")
        self.mVisClosure_W_1jet = self.theFMvisFile.Get("closure_mvis_"+channel+"_1jet_w")
        self.mVisClosure_W_2jet = self.theFMvisFile.Get("closure_mvis_"+channel+"_2jet_w")
        self.mVisClosure_TT = self.theFMvisFile.Get("closure_mvis_"+channel+"_ttmc")        
        #lptclosure
        if self.isDifferential:
            self.lptClosure_W_taupt30to50 = self.theFMvisFile.Get("closure_lpt_taupt30to50_"+channel+"_w")
            self.lptClosure_W_taupt50to70 = self.theFMvisFile.Get("closure_lpt_taupt50to70_"+channel+"_w")
            self.lptClosure_W_tauptgt70 = self.theFMvisFile.Get("closure_lpt_tauptgt70_"+channel+"_w")
            self.lptClosure_QCD_taupt30to50 = self.theFMvisFile.Get("closure_lpt_taupt30to50_"+channel+"_qcd")
            self.lptClosure_QCD_taupt50to70 = self.theFMvisFile.Get("closure_lpt_taupt50to70_"+channel+"_qcd")
            self.lptClosure_QCD_tauptgt70 = self.theFMvisFile.Get("closure_lpt_tauptgt70_"+channel+"_qcd")
            self.lptClosure_TT_taupt30to50 = self.theFMvisFile.Get("closure_lpt_taupt30to50_"+channel+"_ttmc")
            self.lptClosure_TT_taupt50to70 = self.theFMvisFile.Get("closure_lpt_taupt50to70_"+channel+"_ttmc")
            self.lptClosure_TT_tauptgt70 = self.theFMvisFile.Get("closure_lpt_tauptgt70_"+channel+"_ttmc")
            
            self.lptClosure_W_xtrg_taupt30to50 = self.theFMvisFile.Get("closure_lpt_taupt30to50_xtrg_"+channel+"_w")
            self.lptClosure_W_xtrg_taupt50to70 = self.theFMvisFile.Get("closure_lpt_taupt50to70_xtrg_"+channel+"_w")
            self.lptClosure_W_xtrg_tauptgt70 = self.theFMvisFile.Get("closure_lpt_tauptgt70_xtrg_"+channel+"_w")
            self.lptClosure_QCD_xtrg_taupt30to50 = self.theFMvisFile.Get("closure_lpt_taupt30to50_xtrg_"+channel+"_qcd")
            self.lptClosure_QCD_xtrg_taupt50to70 = self.theFMvisFile.Get("closure_lpt_taupt50to70_xtrg_"+channel+"_qcd")
            self.lptClosure_QCD_xtrg_tauptgt70 = self.theFMvisFile.Get("closure_lpt_tauptgt70_xtrg_"+channel+"_qcd")
            self.lptClosure_TT_xtrg_taupt30to50 = self.theFMvisFile.Get("closure_lpt_taupt30to50_xtrg_"+channel+"_ttmc")
            self.lptClosure_TT_xtrg_taupt50to70 = self.theFMvisFile.Get("closure_lpt_taupt50to70_xtrg_"+channel+"_ttmc")
            self.lptClosure_TT_xtrg_tauptgt70 = self.theFMvisFile.Get("closure_lpt_tauptgt70_xtrg_"+channel+"_ttmc")        
        else:
            self.lptClosure_W_taupt30to40 = self.theFMvisFile.Get("closure_lpt_taupt30to40_"+channel+"_w")
            self.lptClosure_W_taupt40to50 = self.theFMvisFile.Get("closure_lpt_taupt40to50_"+channel+"_w")
            self.lptClosure_W_tauptgt50 = self.theFMvisFile.Get("closure_lpt_tauptgt50_"+channel+"_w")
            self.lptClosure_QCD_taupt30to40 = self.theFMvisFile.Get("closure_lpt_taupt30to40_"+channel+"_qcd")
            self.lptClosure_QCD_taupt40to50 = self.theFMvisFile.Get("closure_lpt_taupt40to50_"+channel+"_qcd")
            self.lptClosure_QCD_tauptgt50 = self.theFMvisFile.Get("closure_lpt_tauptgt50_"+channel+"_qcd")
            self.lptClosure_TT_taupt30to40 = self.theFMvisFile.Get("closure_lpt_taupt30to40_"+channel+"_ttmc")
            self.lptClosure_TT_taupt40to50 = self.theFMvisFile.Get("closure_lpt_taupt40to50_"+channel+"_ttmc")
            self.lptClosure_TT_tauptgt50 = self.theFMvisFile.Get("closure_lpt_tauptgt50_"+channel+"_ttmc")
            
            self.lptClosure_W_xtrg_taupt30to40 = self.theFMvisFile.Get("closure_lpt_taupt30to40_xtrg_"+channel+"_w")
            self.lptClosure_W_xtrg_taupt40to50 = self.theFMvisFile.Get("closure_lpt_taupt40to50_xtrg_"+channel+"_w")
            self.lptClosure_W_xtrg_tauptgt50 = self.theFMvisFile.Get("closure_lpt_tauptgt50_xtrg_"+channel+"_w")
            self.lptClosure_QCD_xtrg_taupt30to40 = self.theFMvisFile.Get("closure_lpt_taupt30to40_xtrg_"+channel+"_qcd")
            self.lptClosure_QCD_xtrg_taupt40to50 = self.theFMvisFile.Get("closure_lpt_taupt40to50_xtrg_"+channel+"_qcd")
            self.lptClosure_QCD_xtrg_tauptgt50 = self.theFMvisFile.Get("closure_lpt_tauptgt50_xtrg_"+channel+"_qcd")
            self.lptClosure_TT_xtrg_taupt30to40 = self.theFMvisFile.Get("closure_lpt_taupt30to40_xtrg_"+channel+"_ttmc")
            self.lptClosure_TT_xtrg_taupt40to50 = self.theFMvisFile.Get("closure_lpt_taupt40to50_xtrg_"+channel+"_ttmc")
            self.lptClosure_TT_xtrg_tauptgt50 = self.theFMvisFile.Get("closure_lpt_tauptgt50_xtrg_"+channel+"_ttmc")        
        
        #MET closure? may be in other file?
        self.metClosure_W_0jet = self.theFMvisFile.Get("closure_met_"+channel+"_0jet_w")

        #MT and OSSS closure
        self.theFOSSSClosureFile = ROOT.TFile(theFFDirectory+"FF_QCDcorrectionOSSS.root")
        if self.theFOSSSClosureFile.IsZombie():
            raise RuntimeError("Problem loading the files!")
        
        #self.OSSSClosure_QCD = self.theFOSSSClosureFile.Get("closure_OSSS_mvis_"+channel+"_qcd")
        self.OSSSClosure_QCD = self.theFOSSSClosureFile.Get("closure_OSSS_dr_flat_"+channel+"_qcd")
        self.OSSSClosure_QCD_unc1_up = self.theFOSSSClosureFile.Get("closure_OSSS_mvis_"+channel+"_qcd_unc1_up")
        self.OSSSClosure_QCD_unc1_down = self.theFOSSSClosureFile.Get("closure_OSSS_mvis_"+channel+"_qcd_unc1_down")
        self.OSSSClosure_QCD_unc2_up = self.theFOSSSClosureFile.Get("closure_OSSS_mvis_"+channel+"_qcd_unc2_up")
        self.OSSSClosure_QCD_unc2_down = self.theFOSSSClosureFile.Get("closure_OSSS_mvis_"+channel+"_qcd_unc2_down")

        self.MTClosure_W = self.theFOSSSClosureFile.Get("closure_mt_"+channel+"_w")
        self.MTClosure_W_unc1_up = self.theFOSSSClosureFile.Get("closure_mt_"+channel+"_w_unc1_up")
        self.MTClosure_W_unc1_down = self.theFOSSSClosureFile.Get("closure_mt_"+channel+"_w_unc1_down")
        self.MTClosure_W_unc2_up = self.theFOSSSClosureFile.Get("closure_mt_"+channel+"_w_unc2_up")
        self.MTClosure_W_unc2_down = self.theFOSSSClosureFile.Get("closure_mt_"+channel+"_w_unc2_down")

        #Tau pt corrections
        self.tauPtFile =  ROOT.TFile(theFFDirectory+"tauptcorrection_"+channel+".root")
        self.tauPtCorrection_qcd = self.tauPtFile.Get("mt_0jet_qcd_taupt_iso")
        self.tauPtCorrection_w = self.tauPtFile.Get("mt_0jet_w_taupt_iso")

        #test 0 jet met correction stuff
        if self.attempt0JetMETCorrection:
            self.closure_met_0jet_qcd = self.theFMvisFile.Get('closure_met_'+channel+'_0jet_qcd')

    def get_raw_FF(self,pt,fct):
        return fct.Eval(min(100, pt))

    def get_mvis_closure(self,mvis,fct):
        return fct.Eval(min(250, mvis))

    def get_mt_closure(self,mt, fct):
        return fct.Eval(mt)

    def get_lpt_closure(self,lpt,fct):
        return fct.Eval(min(150, lpt))

    def get_dr_closure(self,dr,fct):
        return fct.Eval(dr)

    def get_MET_closure(self,met,fct):
        return fct.Eval(max(0, met))
        

    def get_ff(self, pt, mt, mvis, lpt, dr, met, njets, xtrg, frac_tt, frac_qcd, frac_w, unc='',upOrDown=''):
        ff_qcd = 1.0
        ff_w = 0
        ff_tt = 1.0    
    
        #Raw ff
        if(njets==0):
            if unc == 'ff_qcd_0jet_unc1':
                if upOrDown == 'up':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_0jet_unc1_up)
                elif upOrDown == 'down':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_0jet_unc1_down)
            elif unc == 'ff_qcd_0jet_unc2':
                if upOrDown == 'up':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_0jet_unc2_up)
                elif upOrDown == 'down':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_0jet_unc2_down)
            else:
                ff_qcd = self.get_raw_FF(pt,self.ff_qcd_0jet)
            if unc == 'ff_w_0jet_unc1':
                if upOrDown == 'up':
                    ff_w=self.get_raw_FF(pt,self.ff_w_0jet_unc1_up)
                elif upOrDown == 'down':
                    ff_w=self.get_raw_FF(pt,self.ff_w_0jet_unc1_down)
            elif unc == 'ff_w_0jet_unc2':                
                if upOrDown == 'up':
                    ff_w=self.get_raw_FF(pt,self.ff_w_0jet_unc2_up)
                elif upOrDown == 'down':
                    ff_w = self.get_raw_FF(pt,self.ff_w_0jet_unc2_down)                    
            else:
                ff_w=self.get_raw_FF(pt,self.ff_w_0jet)
        elif(njets==1):
            #print("raw njets 1 function called")
            if unc == 'ff_qcd_1jet_unc1':
                if upOrDown == 'up':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_1jet_unc1_up)
                elif upOrDown == 'down':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_1jet_unc1_down)
            elif unc == 'ff_qcd_1jet_unc2':
                if upOrDown == 'up':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_1jet_unc2_up)
                elif upOrDown == 'down':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_1jet_unc2_down)
            else:
                ff_qcd=self.get_raw_FF(pt,self.ff_qcd_1jet)
            if unc == 'ff_w_1jet_unc1':
                if upOrDown == 'up':
                    ff_w=self.get_raw_FF(pt,self.ff_w_1jet_unc1_up)
                elif upOrDown == 'down':
                    ff_w=self.get_raw_FF(pt,self.ff_w_1jet_unc1_down)
            elif unc == 'ff_w_1jet_unc2':
                if upOrDown == 'up':
                    ff_w=self.get_raw_FF(pt,self.ff_w_1jet_unc2_up)
                elif upOrDown == 'down':
                    ff_w = self.get_raw_FF(pt,self.ff_w_1jet_unc2_down)
            else:
                ff_w=self.get_raw_FF(pt,self.ff_w_1jet)
        else:
            if unc == 'ff_qcd_2jet_unc1':
                if upOrDown == 'up':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_2jet_unc1_up)
                elif upOrDown == 'down':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_2jet_unc1_down)
            elif unc == 'ff_qcd_2jet_unc2':
                if upOrDown == 'up':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_2jet_unc2_up)
                elif upOrDown == 'down':
                    ff_qcd = self.get_raw_FF(pt,self.ff_qcd_2jet_unc2_down)
            else:
                ff_qcd=self.get_raw_FF(pt,self.ff_qcd_2jet)
            if unc == 'ff_w_2jet_unc1':
                if upOrDown == 'up':
                    ff_w=self.get_raw_FF(pt,self.ff_w_2jet_unc1_up)
                elif upOrDown == 'down':
                    ff_w=self.get_raw_FF(pt,self.ff_w_2jet_unc1_down)
            elif unc == 'ff_w_2jet_unc2':                
                if upOrDown == 'up':
                    ff_w=self.get_raw_FF(pt,self.ff_w_2jet_unc2_up)
                elif upOrDown == 'down':
                    ff_w = self.get_raw_FF(pt,self.ff_w_2jet_unc2_down)
            else:
                ff_w = self.get_raw_FF(pt,self.ff_w_2jet)

        if unc == 'ff_tt_0jet_unc1':
            if upOrDown == 'up':
                ff_tt=self.get_raw_FF(pt,self.ff_tt_0jet_unc1_up)
            elif upOrDown == 'down':
                ff_tt=self.get_raw_FF(pt,self.ff_tt_0jet_unc1_down)
        elif unc == 'ff_tt_0jet_unc2':
            if upOrDown == 'up':
                ff_tt=self.get_raw_FF(pt,self.ff_tt_0jet_unc2_up)
            elif upOrDown == 'down':
                ff_tt=self.get_raw_FF(pt,self.ff_tt_0jet_unc2_down)
        else:
            ff_tt = self.get_raw_FF(pt,self.ff_tt_0jet)

        #MET closure
        """
        if njets == 0:                        
            #print("MET closure")
            #print(self.get_MET_closure(met,self.metClosure_W_0jet))
            ff_w = ff_w*self.get_MET_closure(met,self.metClosure_W_0jet)
        """
        
        #lpt closures
        if self.isDifferential:
            if (xtrg):            
                if unc == 'lptclosure_xtrg_qcd':
                    if pt > 30 and pt <= 50:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt30to50)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt30to50)*0.9
                    elif pt > 50 and pt <= 70:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt50to70)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt50to70)*0.9
                    elif pt > 70:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_tauptgt70)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_tauptgt70)*0.9
                else:
                    if pt > 30 and pt <= 50:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt30to50)
                    elif pt > 50 and pt <= 70:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt50to70)
                    elif pt > 70:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_tauptgt70)
                if unc == 'lptclosure_xtrg_w':
                    if pt > 30 and pt <= 50:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt30to50)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt30to50)*0.9
                    elif pt > 50 and pt <= 70:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt50to70)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt50to70)*0.9
                    elif pt > 70:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_tauptgt70)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_tauptgt70)*0.9
                else:
                    if pt > 30 and pt <= 50:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt30to50)
                    elif pt > 50 and pt <= 70:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt50to70)
                    elif pt > 70:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_tauptgt70)
                if unc == 'lptclosure_xtrg_tt':                
                    if pt > 30 and pt <= 50:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt30to50)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt30to50)*0.9
                    elif pt > 50 and pt <= 70:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt50to70)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt50to70)*0.9
                    elif pt > 70:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_tauptgt70)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_tauptgt70)*0.9
                else:
                    if pt > 30 and pt <= 50:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt30to50)
                    elif pt > 50 and pt <= 70:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt50to70)
                    elif pt > 70:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_tauptgt70)
            else:
                if unc == 'lptclosure_qcd':
                    if pt > 30 and pt <= 50:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt30to50)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt30to50)*0.9
                    elif pt > 50 and pt <= 70:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt50to70)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt50to70)*0.9
                    elif pt > 70:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_tauptgt70)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_tauptgt70)*0.9
                else:
                    if pt > 30 and pt <= 50:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt30to50)
                    elif pt > 50 and pt <= 70:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt50to70)
                    elif pt > 70:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_tauptgt70)
                if unc == 'lptclosure_w':
                    if pt > 30 and pt <= 50:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt30to50)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt30to50)*0.9
                    elif pt > 50 and pt <= 70:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt50to70)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt50to70)*0.9
                    elif pt > 70:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_tauptgt70)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_tauptgt70)*0.9
                else:
                    if pt > 30 and pt <= 50:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt30to50)
                    elif pt > 50 and pt <= 70:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt50to70)
                    elif pt > 70:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_tauptgt70)
                if unc == 'lptclosure_tt':
                    if pt > 30 and pt <= 50:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt30to50)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt30to50)*0.9
                    elif pt > 50 and pt <= 70:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt50to70)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt50to70)*0.9
                    elif pt > 70:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_tauptgt70)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_tauptgt70)*0.9
                else:
                    if pt > 30 and pt <= 50:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt30to50)
                    elif pt > 50 and pt <= 70:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt50to70)
                    elif pt > 70:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_tauptgt70)
        else:
            if (xtrg):            
                if unc == 'lptclosure_xtrg_qcd':
                    if pt > 30 and pt <= 40:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt30to40)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt30to40)*0.9
                    elif pt > 40 and pt <= 50:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt40to50)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt40to50)*0.9
                    elif pt > 50:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_tauptgt50)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_tauptgt50)*0.9
                else:
                    if pt > 30 and pt <= 40:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt30to40)
                    elif pt > 40 and pt <= 50:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_taupt40to50)
                    elif pt > 50:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_xtrg_tauptgt50)
                if unc == 'lptclosure_xtrg_w':
                    if pt > 30 and pt <= 40:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt30to40)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt30to40)*0.9
                    elif pt > 40 and pt <= 50:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt40to50)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt40to50)*0.9
                    elif pt > 50:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_tauptgt50)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_tauptgt50)*0.9
                else:
                    if pt > 30 and pt <= 40:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt30to40)
                    elif pt > 40 and pt <= 50:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_taupt40to50)
                    elif pt > 50:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_xtrg_tauptgt50)
                if unc == 'lptclosure_xtrg_tt':                
                    if pt > 30 and pt <= 40:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt30to40)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt30to40)*0.9
                    elif pt > 40 and pt <= 50:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt40to50)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt40to50)*0.9
                    elif pt > 50:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_tauptgt50)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_tauptgt50)*0.9
                else:
                    if pt > 30 and pt <= 40:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt30to40)
                    elif pt > 40 and pt <= 50:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_taupt40to50)
                    elif pt > 50:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_xtrg_tauptgt50)
            else:
                if unc == 'lptclosure_qcd':
                    if pt > 30 and pt <= 40:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt30to40)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt30to40)*0.9
                    elif pt > 40 and pt <= 50:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt40to50)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt40to50)*0.9
                    elif pt > 50:
                        if upOrDown == "up":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_tauptgt50)*1.1
                        elif upOrDown == "down":
                            ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_tauptgt50)*0.9
                else:
                    if pt > 30 and pt <= 40:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt30to40)
                    elif pt > 40 and pt <= 50:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_taupt40to50)
                    elif pt > 50:
                        ff_qcd = ff_qcd*self.get_lpt_closure(lpt, self.lptClosure_QCD_tauptgt50)
                if unc == 'lptclosure_w':
                    if pt > 30 and pt <= 40:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt30to40)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt30to40)*0.9
                    elif pt > 40 and pt <= 50:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt40to50)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt40to50)*0.9
                    elif pt > 50:
                        if upOrDown == "up":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_tauptgt50)*1.1
                        elif upOrDown == "down":
                            ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_tauptgt50)*0.9
                else:
                    if pt > 30 and pt <= 40:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt30to40)
                    elif pt > 40 and pt <= 50:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_taupt40to50)
                    elif pt > 50:
                        ff_w = ff_w*self.get_lpt_closure(lpt, self.lptClosure_W_tauptgt50)
                if unc == 'lptclosure_tt':
                    if pt > 30 and pt <= 40:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt30to40)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt30to40)*0.9
                    elif pt > 40 and pt <= 50:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt40to50)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt40to50)*0.9
                    elif pt > 50:
                        if upOrDown == "up":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_tauptgt50)*1.1
                        elif upOrDown == "down":
                            ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_tauptgt50)*0.9
                else:
                    if pt > 30 and pt <= 40:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt30to40)
                    elif pt > 40 and pt <= 50:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_taupt40to50)
                    elif pt > 50:
                        ff_tt = ff_tt*self.get_lpt_closure(lpt, self.lptClosure_TT_tauptgt50)
        
        #MT and OSSS corrections
        if unc == 'mtclosure_w_unc1':
            if upOrDown == 'up':
                ff_w = ff_w*self.get_mt_closure(mt,self.MTClosure_W_unc1_up)
            elif upOrDown == 'down':
                ff_w = ff_w*self.get_mt_closure(mt,self.MTClosure_W_unc1_down)
        elif unc == 'mtclosure_w_unc2':            
            if upOrDown == 'up':
                ff_w = ff_w*self.get_mt_closure(mt,self.MTClosure_W_unc2_up)
            elif upOrDown == 'down':
                ff_w = ff_w*self.get_mt_closure(mt,self.MTClosure_W_unc2_down)
        else:
            ff_w = ff_w*self.get_mt_closure(mt,self.MTClosure_W)
        
        if unc == 'osssclosure_qcd':            
            if upOrDown == 'up':                
                ff_qcd = ff_qcd * self.get_dr_closure(dr,self.OSSSClosure_QCD)*1.1
            elif upOrDown == 'down':
                ff_qcd = ff_qcd * self.get_dr_closure(dr,self.OSSSClosure_QCD)*0.9                       
        else:                        
            ff_qcd = ff_qcd * self.get_dr_closure(dr,self.OSSSClosure_QCD)
 
        #put in this test correction
        #if self.attempt0JetMETCorrection:
            #ff_qcd = ff_qcd * self.get_MET_closure(met,self.closure_met_0jet_qcd)
            #if met <70:
                #ff_w = ff_w * 1.05
            #elif met > 70 and met < 100:
                #ff_w = ff_w * 0.95
            #else:
                #ff_w = ff_w * 0.85
        
        ff_cmb = frac_tt*ff_tt + frac_qcd*ff_qcd + frac_w*ff_w
        return ff_cmb
