# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
import argparse, os, sys, os.path, json
import math
#reload(sys)
#sys.setdefaultencoding('utf8')
import simpleamt
from boto.mturk.price import Price
from boto.mturk.question import HTMLQuestion
from boto.mturk.connection import MTurkRequestError
import io
from exp_util import *

def load_address_dataset():
    file_path = 'examples/address/address_dataset'
   
    addr = list() 
    label = dict()
    with open(file_path, 'r') as f:
        for addr_ in f:
            addr_ = addr_.replace('ppart','part')
            tokens = addr_.split(' ')
            addr.append( ' '.join(tokens[1:]) ) 
            
            if '*' in tokens[0]:
                label[addr[-1]] = 1
            else:
                label[addr[-1]] = 0
    return addr, label


def update_address_template(addr):
    assert len(addr) == 20

    html_str = """
    <html>
      <head>
        <script src='//cdnjs.cloudflare.com/ajax/libs/jquery/2.1.1/jquery.min.js'></script>
        <script src='//cdnjs.cloudflare.com/ajax/libs/json3/3.3.2/json3.min.js'></script>
        <link href='//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css' rel='stylesheet'>    
      </head>
      <body>
        <!-- Instructions -->
        <div class='row'>
        <div class='col-xs-12 col-md-12'>
        <div class='panel panel-primary'><a class='panel-heading' href='javascript:void(0);' id='collapseTrigger'><strong>Home Address in Oregon Verification Instructions</strong></a>
        <div class='panel-body' id='instructionBody'>
        <p>Verify the address below as a home address in Oregon, U.S.A.</p>
        <ul>
            <li>Indicate if the address (below) is a valid home address in Oregon, U.S.A. or not</li>
            <li><strong>A valid home adsdress</strong> should have street number and name, (optional) a unit number, and city (e.g., Portland), state (e.g., OR) and ZIP (e.g., 97007),</li> 
            <li>and all components strictly <strong>in this order</strong>; a unit number can't come before street address.</li>
            <li> e.g., 12240 Southwest Horizon Boulevard Appartment # 106 Portland, OR 97007 <strong> is valid</strong>
            <li> e.g., 12240 Southwest Horizon Boulevard Portland, OR 97007 <strong> is valid</strong>
            <li> e.g., Appartment # 106 12240 Southwest Horizon Boulevard Portland, OR 97007 <strong>is not valid</strong>
            <li> e.g., 12240 Southwest Horizon Boulevard, OR 97007 <strong> is not valid</strong>
        </ul>
        </div>
        </div>
        </div>
        </div>
        <!-- End Instructions -->

        <div class='container'>
          <h1>20 Home Addresses in Oregon Verification</h1>

          <p>PRACTICE 1) Street number must come before unit number:\nAppartment # 106 12240 Southwest Horizon Boulevard Portland, OR 97007</p>
          <div class='form-group'><label class='verify00'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v00_option1' name='isAddressValid00' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v00_option2' name='isAddressValid00' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>PRACTICE 2) APT/UNIT number is optional & Portland, OR 97007 must be present:\n12240 Southwest Horizon Boulevard Portland, OR 97007</p>
          <div class='form-group'><label class='verify01'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v01_option1' name='isAddressValid01' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v01_option2' name='isAddressValid01' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
        </div>

        <div class='container'>
          <p>%s</p>
          <div class='form-group'><label class='verify1'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v1_option1' name='isAddressValid' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v1_option2' name='isAddressValid' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify2'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v2_option1' name='isAddressValid2' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v2_option2' name='isAddressValid2' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify3'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v3_option1' name='isAddressValid3' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v3_option2' name='isAddressValid3' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify4'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v4_option1' name='isAddressValid4' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v4_option2' name='isAddressValid4' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify5'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v5_option1' name='isAddressValid5' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v5_option2' name='isAddressValid5' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify6'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v6_option1' name='isAddressValid6' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v6_option2' name='isAddressValid6' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify7'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v7_option1' name='isAddressValid7' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v7_option2' name='isAddressValid7' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify8'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v8_option1' name='isAddressValid8' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v8_option2' name='isAddressValid8' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify9'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v9_option1' name='isAddressValid9' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v9_option2' name='isAddressValid9' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify10'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v10_option1' name='isAddressValid10' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v10_option2' name='isAddressValid10' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify11'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v11_option1' name='isAddressValid11' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v11_option2' name='isAddressValid11' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify12'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v12_option1' name='isAddressValid12' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v12_option2' name='isAddressValid12' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify13'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v13_option1' name='isAddressValid13' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v13_option2' name='isAddressValid13' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify14'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v14_option1' name='isAddressValid14' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v14_option2' name='isAddressValid14' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify15'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v15_option1' name='isAddressValid15' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v15_option2' name='isAddressValid15' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify16'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v16_option1' name='isAddressValid16' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v16_option2' name='isAddressValid16' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify17'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v17_option1' name='isAddressValid17' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v17_option2' name='isAddressValid17' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify18'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v18_option1' name='isAddressValid18' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v18_option2' name='isAddressValid18' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify19'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v19_option1' name='isAddressValid19' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v19_option2' name='isAddressValid19' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify20'>Is the address valid?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v20_option1' name='isAddressValid20' required='' type='radio' value='yes' /> Is valid </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v20_option2' name='isAddressValid20' required='' type='radio' value='no' /> Is NOT valid </label></div>
          </div>
          <div><br></div>
        </div>

        <style type='text/css'>#collapseTrigger{
          color:#00F;
          display: block;
          text-decoration: none;
        }
        #submitButton{
          white-space: normal;
        }
        .image{
          margin-bottom: 15px;
        }
        .group-label{
          display: block;
        }
        .radio-inline>label{
          font-weight: normal;
        }
        </style>
        """%(addr[0],addr[1],addr[2],addr[3],addr[4],addr[5],addr[6],addr[7],addr[8],addr[9],
             addr[10],addr[11],addr[12],addr[13],addr[14],addr[15],addr[16],addr[17],addr[18],addr[19])

    html_str2 = """
        {% include 'hit_templates/simpleamt.html' %}

        <script>
          $(function() {

            function main() {

              // If the HIT is not in preview mode, then we need to enable the UI
              // and set up the logic for submitting.
              if (!simpleamt.isPreview()) {
                enable_ui();

                // You need to call this in every HIT; if you forget then you will
                // get an error message when you try and submit the HIT.
                simpleamt.setupSubmit();

                // Set up a click handler for the submit button.
                // Typically this will validate user output and either submit the
                // HIT if the data is good or show an error message to the user if
                // the data is bad. If this click handler returns false then the
                // HIT will not be submitted.
                // WARNING: If the click handler throws an exception
                // then by default the HIT will be submitted; this is a fertile
                // source of bugs.
                $('#submit-btn').click(function() {
                  // Read in the examples.
                  var v00_is_valid = '';
                  if (document.getElementById('v00_option1').checked) {
                    v00_is_valid = document.getElementById('v00_option1').value;
                  }
                  else if (document.getElementById('v00_option2').checked) {
                    v00_is_valid = document.getElementById('v00_option2').value;
                  }
                  var v01_is_valid = '';
                  if (document.getElementById('v01_option1').checked) {
                    v01_is_valid = document.getElementById('v01_option1').value;
                  }
                  else if (document.getElementById('v01_option2').checked) {
                    v01_is_valid = document.getElementById('v01_option2').value;
                  }

                  // Read in the verification result.
                  var v1_is_valid = '';
                  if (document.getElementById('v1_option1').checked) {
                    v1_is_valid = document.getElementById('v1_option1').value;
                  }
                  else if (document.getElementById('v1_option2').checked) {
                    v1_is_valid = document.getElementById('v1_option2').value;
                  }
                  var v2_is_valid = '';
                  if (document.getElementById('v2_option1').checked) {
                    v2_is_valid = document.getElementById('v2_option1').value;
                  }
                  else if (document.getElementById('v2_option2').checked) {
                    v2_is_valid = document.getElementById('v2_option2').value;
                  }
                  var v3_is_valid = '';
                  if (document.getElementById('v3_option1').checked) {
                    v3_is_valid = document.getElementById('v3_option1').value;
                  }
                  else if (document.getElementById('v3_option2').checked) {
                    v3_is_valid = document.getElementById('v3_option2').value;
                  }
                  var v4_is_valid = '';
                  if (document.getElementById('v4_option1').checked) {
                    v4_is_valid = document.getElementById('v4_option1').value;
                  }
                  else if (document.getElementById('v4_option2').checked) {
                    v4_is_valid = document.getElementById('v4_option2').value;
                  }
                  var v5_is_valid = '';
                  if (document.getElementById('v5_option1').checked) {
                    v5_is_valid = document.getElementById('v5_option1').value;
                  }
                  else if (document.getElementById('v5_option2').checked) {
                    v5_is_valid = document.getElementById('v5_option2').value;
                  }
                  var v6_is_valid = '';
                  if (document.getElementById('v6_option1').checked) {
                    v6_is_valid = document.getElementById('v6_option1').value;
                  }
                  else if (document.getElementById('v6_option2').checked) {
                    v6_is_valid = document.getElementById('v6_option2').value;
                  }
                  var v7_is_valid = '';
                  if (document.getElementById('v7_option1').checked) {
                    v7_is_valid = document.getElementById('v7_option1').value;
                  }
                  else if (document.getElementById('v7_option2').checked) {
                    v7_is_valid = document.getElementById('v7_option2').value;
                  }
                  var v8_is_valid = '';
                  if (document.getElementById('v8_option1').checked) {
                    v8_is_valid = document.getElementById('v8_option1').value;
                  }
                  else if (document.getElementById('v8_option2').checked) {
                    v8_is_valid = document.getElementById('v8_option2').value;
                  }
                  var v9_is_valid = '';
                  if (document.getElementById('v9_option1').checked) {
                    v9_is_valid = document.getElementById('v9_option1').value;
                  }
                  else if (document.getElementById('v9_option2').checked) {
                    v9_is_valid = document.getElementById('v9_option2').value;
                  }
                  var v10_is_valid = '';
                  if (document.getElementById('v10_option1').checked) {
                    v10_is_valid = document.getElementById('v10_option1').value;
                  }
                  else if (document.getElementById('v10_option2').checked) {
                    v10_is_valid = document.getElementById('v10_option2').value;
                  }
                  var v11_is_valid = '';
                  if (document.getElementById('v11_option1').checked) {
                    v11_is_valid = document.getElementById('v11_option1').value;
                  }
                  else if (document.getElementById('v11_option2').checked) {
                    v11_is_valid = document.getElementById('v11_option2').value;
                  }
                  var v12_is_valid = '';
                  if (document.getElementById('v12_option1').checked) {
                    v12_is_valid = document.getElementById('v12_option1').value;
                  }
                  else if (document.getElementById('v12_option2').checked) {
                    v12_is_valid = document.getElementById('v12_option2').value;
                  }
                  var v13_is_valid = '';
                  if (document.getElementById('v13_option1').checked) {
                    v13_is_valid = document.getElementById('v13_option1').value;
                  }
                  else if (document.getElementById('v13_option2').checked) {
                    v13_is_valid = document.getElementById('v13_option2').value;
                  }
                  var v14_is_valid = '';
                  if (document.getElementById('v14_option1').checked) {
                    v14_is_valid = document.getElementById('v14_option1').value;
                  }
                  else if (document.getElementById('v14_option2').checked) {
                    v14_is_valid = document.getElementById('v14_option2').value;
                  }
                  var v15_is_valid = '';
                  if (document.getElementById('v15_option1').checked) {
                    v15_is_valid = document.getElementById('v15_option1').value;
                  }
                  else if (document.getElementById('v15_option2').checked) {
                    v15_is_valid = document.getElementById('v15_option2').value;
                  }
                  var v16_is_valid = '';
                  if (document.getElementById('v16_option1').checked) {
                    v16_is_valid = document.getElementById('v16_option1').value;
                  }
                  else if (document.getElementById('v16_option2').checked) {
                    v16_is_valid = document.getElementById('v16_option2').value;
                  }
                  var v17_is_valid = '';
                  if (document.getElementById('v17_option1').checked) {
                    v17_is_valid = document.getElementById('v17_option1').value;
                  }
                  else if (document.getElementById('v17_option2').checked) {
                    v17_is_valid = document.getElementById('v17_option2').value;
                  }
                  var v18_is_valid = '';
                  if (document.getElementById('v18_option1').checked) {
                    v18_is_valid = document.getElementById('v18_option1').value;
                  }
                  else if (document.getElementById('v18_option2').checked) {
                    v18_is_valid = document.getElementById('v18_option2').value;
                  }
                  var v19_is_valid = '';
                  if (document.getElementById('v19_option1').checked) {
                    v19_is_valid = document.getElementById('v19_option1').value;
                  }
                  else if (document.getElementById('v19_option2').checked) {
                    v19_is_valid = document.getElementById('v19_option2').value;
                  }
                  var v20_is_valid = '';
                  if (document.getElementById('v20_option1').checked) {
                    v20_is_valid = document.getElementById('v20_option1').value;
                  }
                  else if (document.getElementById('v20_option2').checked) {
                    v20_is_valid = document.getElementById('v20_option2').value;
                  }

                  // Construct an object containing the output of this assignment.
                  var output = {
                    v1_is_valid: v1_is_valid,
                    v2_is_valid: v2_is_valid,
                    v3_is_valid: v3_is_valid,
                    v4_is_valid: v4_is_valid,
                    v5_is_valid: v5_is_valid,
                    v6_is_valid: v6_is_valid,
                    v7_is_valid: v7_is_valid,
                    v8_is_valid: v8_is_valid,
                    v9_is_valid: v9_is_valid,
                    v10_is_valid: v10_is_valid,
                    v11_is_valid: v11_is_valid,
                    v12_is_valid: v12_is_valid,
                    v13_is_valid: v13_is_valid,
                    v14_is_valid: v14_is_valid,
                    v15_is_valid: v15_is_valid,
                    v16_is_valid: v16_is_valid,
                    v17_is_valid: v17_is_valid,
                    v18_is_valid: v18_is_valid,
                    v19_is_valid: v19_is_valid,
                    v20_is_valid: v20_is_valid
                  };

                  // Sanity check
                  if (v00_is_valid === 'yes' ||
                      v01_is_valid === 'no') {
                    alert('Please read the instructions carefully and try to answer the questions correctly.');
                    return false; }

                  // Validate the output
                  if (output.v1_is_valid.length < 1 ||
                      output.v2_is_valid.length < 1 ||
                      output.v3_is_valid.length < 1 ||
                      output.v4_is_valid.length < 1 ||
                      output.v5_is_valid.length < 1 ||
                      output.v6_is_valid.length < 1 ||
                      output.v7_is_valid.length < 1 ||
                      output.v8_is_valid.length < 1 ||
                      output.v9_is_valid.length < 1 ||
                      output.v10_is_valid.length < 1 ||
                      output.v11_is_valid.length < 1 ||
                      output.v12_is_valid.length < 1 ||
                      output.v13_is_valid.length < 1 ||
                      output.v14_is_valid.length < 1 ||
                      output.v15_is_valid.length < 1 ||
                      output.v16_is_valid.length < 1 ||
                      output.v17_is_valid.length < 1 ||
                      output.v18_is_valid.length < 1 ||
                      output.v19_is_valid.length < 1 ||
                      output.v20_is_valid.length < 1) {
                    alert('Please verify all the addresses to submit');
                    return false; }
                  else {
                    simpleamt.setOutput(output); }
                })
              }
            }

            function enable_ui() {
              // Enable the submit button. You must do this in every HIT.
              $('#submit-btn').prop('disabled', false);
            }

            main();

          });
        </script>

      </body>
    </html>
    """
                           
    html_str = html_str + html_str2
    html_str = html_str.encode('ascii', 'ignore')
    with open('hit_templates/address.html','w') as file:
        file.write(html_str)

def triangular_walk(n_workers=1000, n_max=50, sandbox=True):
    addr, label = load_address_dataset()
    n_items = len(addr)

    walks = dict() # walks[i] = (record_id, n, k)
    completed = set() # keeps track of completed triangles for batch item replacement
    estimates = dict() # estimates[n_worker_] = #error_estimate
    linear_estimates = dict()

    batch_size = 20 # this is fixed
    for i in range(batch_size):
        walks[i] = (np.random.choice(n_items), 0., 0.)
        linear_estimates[i] = list()
    batch = []
    for k, v in walks.iteritems():
        batch.append(addr[v[0]])

    n_workers_ = n_workers
    while n_workers_ > 0:
        update_address_template(batch)
        try: os.remove('examples/address/hit_ids/txt')
        except OSError: pass

        hit_ids = launch_hit(sandbox=sandbox,
                    hit_template_path='hit_templates/address.html',
                    hit_properties_path='hit_properties/address.json',
                    hit_ids_path='examples/address/hit_ids.txt') # post a random batch task

        hits = get_results(hit_ids,sandbox=sandbox) # for now, results is a singleton list
        with open('examples/address/results.txt','a+') as f:
            for hit in hits:
                hit['batch'] = batch
                f.write(json.dumps(hit)+'\n')
        disable_hit(hit_ids,sandbox=sandbox) # remove the HITs on hit_ids.txt
        print 'n_workers %s'%(n_workers-n_workers_+1)

        if len(hits) == 0:
            continue
        
        # this allows multiple assignments per hit
        # if we restrict a single assignment per hit, then we can pretend that the for loop is not here.
        for hit in hits: 
            for i in range(batch_size):
                record_id = walks[i][0]
                l_ = label[addr[record_id]]
                n_ = walks[i][1] + 1.        
                k_ = walks[i][2]

                resp = hit['output']['v%s_is_valid'%(i+1)]
                if resp == 'no': 
                    k_ += 1

                # check for stopping conditions
                if n_ < n_max and k_/n_ > 0.5:
                    walks[i] = (walks[i][0], n_, k_)
                else:
                    completed.add(i)
                    if k_/n_ <= 0.5:
                        linear_estimates[i].append(0.)
                    else:
                        if (2-n_max-2*k_)**2 -4*(2*n_max-2)*k_ >= 0:
                            p_ = ( 2.*k_+n_max-2+math.sqrt((2-n_max-2*k_)**2-4*(2*n_max-2)*k_)) / (4.*n_max-4)
                        else:
                            p_ = ((2.*k_+n_max-2)/(4*n_max-4))
                        #p_ = k_/n_max
                        #print 1./(2*p_-1.)
                        p_ = max(p_, 0.6)
                        linear_estimates[i].append(1./(2*p_-1.) - 0.165*p_*(1-p_)/(2*p_ - 1)**2)    

            for i in completed:
                walks[i] = (np.random.choice(n_items), 0., 0.)
                batch[i] = addr[walks[i][0]]

            cur_estimates = list()

            for i in range(batch_size):
                if len(linear_estimates[i]) > 0:
                    cur_estimates.append(np.mean(linear_estimates[i]))

            if len(cur_estimates) > 0:
                estimates[n_workers-n_workers_+1] = np.mean(cur_estimates) * n_items
            n_workers_ -= 1
            completed = set()

    return estimates
        

def test_exp_address_dataset():
    # load address dataset
    addr, label = load_address_dataset()
    batch = create_batch(addr,batch_size=20)
    update_address_template(batch)

    # for HTML/javascript testing purposes.
    # Open the rendered page locally using python -m SimpleHTTPServer 8080
    os.system('python render_template.py \
                --html_template=hit_templates/address.html \
                --rendered_html=rendered_templates/address.html')

    # hit_ids.txt is used by the simpleamt APIs
    try:
        os.remove('examples/address/hit_ids.txt')
    except OSError:
        pass

    hit_ids = launch_hit(
                hit_template_path='hit_templates/address.html',
                hit_properties_path='hit_properties/address.json',
                hit_ids_path='examples/address/hit_ids.txt') # post a random batch task

    #get results until we have some results back?
    results = get_results(hit_ids)
    with open('examples/address/results.txt','a+') as f:
        for r in results:
            f.write(json.dumps(r))
    # approve submitted HITs
    disable_hit(hit_ids) # remove the HITs on hit_ids.txt

def test_triangular_walk():
    import time, pickle
    start_time = time.time()
    #addr, label = load_address_dataset()
    #estimates = triangular_walk_using_amt(addr, label, update_address_template, 
    #                'examples/address/hit_ids.txt','hit_templates/address.html',
    #                'hit_properties/address.json', 'examples/address/results.txt',
    #                n_workers=1000, sandbox=False)
    estimates = triangular_walk(n_workers=500, sandbox=False)
    end_time = time.time()
    pickle.dump(estimates, open('examples/address/t_walk.p','wb'))
    print('It took %s.'%(end_time - start_time))
    
def parse_results_file():
    import re
    result_file = 'examples/address/results.txt'
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            positions = [m.start() for m in re.finditer('{"assignment_id', line)]
            offset = 0
            for p in positions:
                line = line[:p+offset] + '\n' + line[p+offset:]
                offset += 1
            results += line.split('\n')
    for r in results:
        if len(r.strip()) == 0: continue
        test = json.loads(r)
        test['tmp'] = 'hi'
        print test

def process_estimates():
    import pickle
    estimates = pickle.load(open('examples/address/t_walk.p','rb'))

    w_range = np.arange(10,500,10)
    X, Y, GT = list(), list(), list()
    for w in w_range:
        X.append(w)
        Y.append(estimates[w])
        GT.append(90)

    with open('examples/address/temp.txt','w') as f:
        for i in range(len(w_range)):
            f.write(str(X[i]) + ',' + str(Y[i]) + ',' + str(GT[i]) +'\n')

    '''
    fig, ax = plt.subplots(1,figsize=(8,5))
    colors = ['#00ff99','#0099ff','#ffcc00','#ff5050','#9900cc','#5050ff','#99cccc','#0de4f6']
    markers = ['o-','v-','^-','s-','*-','x-','+-','D-']
    shapes = ['--','-*']

    ax.plot(X, GT, '--', linewidth=2.5, color="#333333",label="GT")
    ax.plot(X, Y, 'o-', linewidth=2.5, color='#00ff99',label="T-WALK(50)")
    ax.gird()
    ax.legend(prop={'size':15}).get_frame().set_alpha(0.5)
    fig.savefig('examples/address/t_walk_fig.png', bbox_inches='tight')
    '''

if __name__ == '__main__':
    #load_address_dataset()
    #test_exp_address_dataset()
    #test_triangular_walk()
    process_estimates()
