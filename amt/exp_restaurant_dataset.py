# -*- coding: utf-8 -*-
import numpy as np
import argparse, os, sys, os.path, csv, json
import math
import pickle
#reload(sys)
#sys.setdefaultencoding('utf8')
import simpleamt
from boto.mturk.price import Price
from boto.mturk.question import HTMLQuestion
from boto.mturk.connection import MTurkRequestError
import io
from exp_util import *


def load_restaurant_dataset():
    file_path = 'examples/restaurant/restaurant.csv'

    records = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            gid = str(row[1])
            rid = str(row[0])
            name = row[2] + ', ' + row[3] + ', ' + row[4] + ', ' + row[5]
            records[rid] = (gid, name)

    rest = list()
    label = dict()
    hard_pairs = pickle.load(open('examples/restaurant/hard_pairs.p','rb'))
    for p in hard_pairs:
        candidate = 'Address 1) %s<br /> Address 2) %s'%(records[p[0][0]][1], records[p[0][1]][1])
        rest.append(candidate)
        if records[p[0][0]][0] == records[p[0][1]][0]:
            label[rest[-1]] = 1
        else:
            label[rest[-1]] = 0

    return rest, label
    

def update_restaurant_template(addr):
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
        <div class='panel panel-primary'><a class='panel-heading' href='javascript:void(0);' id='collapseTrigger'><strong>Instructions</strong></a>
        <div class='panel-body' id='instructionBody'>
        <p>Restaurant addresses de-duplication</p>
        <ul>
            <li>Check if the two addresses point to the same restaurant in CA.</li>
            <li>Restaurant address records are in a form of 'name, address, city, food type':<br />e.g.) arnie morton's of chicago, 435 s. la cienega blv., los angeles, american 
            <li> An example of two address records pointing to the same restaurant:<br />a)art's delicatessen, 12224 ventura blvd., studio city, american<br />b)art's deli, 12224 ventura blvd., studio city, delis
            <li> An example of two address records not belonging to the same restaurant:<br />a)the palm, 9001 santa monica blvd., los angeles, american<br />b)patina, 5955 melrose ave., los angeles, californian
        </ul>
        </div>
        </div>
        </div>
        </div>
        <!-- End Instructions -->

        <div class='container'>
          <h1>20 Restaurant address records de-duplication</h1>

          <p>PRACTICE 1)<br />Address 1) art's delicatessen, 12224 ventura blvd., studio city, american<br />Address 2) art's deli, 12224 ventura blvd., studio city, delis
</p>
          <div class='form-group'><label class='verify00'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v00_option1' name='isAddressValid00' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v00_option2' name='isAddressValid00' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>PRACTICE 2)<br />Address 1) the palm, 9001 santa monica blvd., los angeles, american<br />Address 2) patina, 5955 melrose ave., los angeles, californian
</p>
          <div class='form-group'><label class='verify01'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v01_option1' name='isAddressValid01' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v01_option2' name='isAddressValid01' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
        </div>

        <div class='container'>
          <p>%s</p>
          <div class='form-group'><label class='verify1'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v1_option1' name='isAddressValid' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v1_option2' name='isAddressValid' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify2'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v2_option1' name='isAddressValid2' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v2_option2' name='isAddressValid2' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify3'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v3_option1' name='isAddressValid3' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v3_option2' name='isAddressValid3' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify4'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v4_option1' name='isAddressValid4' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v4_option2' name='isAddressValid4' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify5'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v5_option1' name='isAddressValid5' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v5_option2' name='isAddressValid5' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify6'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v6_option1' name='isAddressValid6' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v6_option2' name='isAddressValid6' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify7'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v7_option1' name='isAddressValid7' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v7_option2' name='isAddressValid7' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify8'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v8_option1' name='isAddressValid8' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v8_option2' name='isAddressValid8' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify9'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v9_option1' name='isAddressValid9' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v9_option2' name='isAddressValid9' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify10'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v10_option1' name='isAddressValid10' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v10_option2' name='isAddressValid10' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify11'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v11_option1' name='isAddressValid11' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v11_option2' name='isAddressValid11' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify12'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v12_option1' name='isAddressValid12' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v12_option2' name='isAddressValid12' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify13'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v13_option1' name='isAddressValid13' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v13_option2' name='isAddressValid13' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify14'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v14_option1' name='isAddressValid14' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v14_option2' name='isAddressValid14' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify15'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v15_option1' name='isAddressValid15' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v15_option2' name='isAddressValid15' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify16'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v16_option1' name='isAddressValid16' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v16_option2' name='isAddressValid16' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify17'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v17_option1' name='isAddressValid17' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v17_option2' name='isAddressValid17' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify18'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v18_option1' name='isAddressValid18' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v18_option2' name='isAddressValid18' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify19'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v19_option1' name='isAddressValid19' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v19_option2' name='isAddressValid19' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
          </div>
          <div><br></div>
          <p>%s</p>
          <div class='form-group'><label class='verify20'>Are these addresses belong to the same restaurant?</label>
            <div class='radio-inline'><label><input autocomplete='off' id='v20_option1' name='isAddressValid20' required='' type='radio' value='yes' /> Belong to the same </label></div>
            <div class='radio-inline'><label><input autocomplete='off' id='v20_option2' name='isAddressValid20' required='' type='radio' value='no' /> Doesn't belong to the same </label></div>
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
                  if (v00_is_valid === 'no' ||
                      v01_is_valid === 'yes') {
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
    with open('hit_templates/restaurant.html','w') as file:
        file.write(html_str)


def test_exp_restaurant_dataset():
    # load address dataset
    data, label = load_restaurant_dataset()
    batch = create_batch(data,batch_size=20)
    update_restaurant_template(batch)

    # for HTML/javascript testing purposes.
    # Open the rendered page locally using python -m SimpleHTTPServer 8080
    os.system('python render_template.py \
                --html_template=hit_templates/restaurant.html \
                --rendered_html=rendered_templates/restaurant.html')

    # hit_ids.txt is used by the simpleamt APIs
    try:
        os.remove('examples/restaurant/hit_ids.txt')
    except OSError:
        pass

    hit_ids = launch_hit(
                hit_template_path='hit_templates/restaurant.html',
                hit_properties_path='hit_properties/restaurant.json',
                hit_ids_path='examples/restaurant/hit_ids.txt') # post a random batch task

    # get results until we have some results back?
    results = get_results(hit_ids)

    with open('examples/restaurant/results.txt','a+') as f:
        for r in results:
            f.write(json.dumps(r))

    # approve submitted HITs
    disable_hit(hit_ids) # remove the HITs on hit_ids.txt

def test_triangular_walk():
    import time, pickle
    start_time = time.time()
    data, label = load_restaurant_dataset()
    estimates = triangular_walk_using_amt(data, label, update_restaurant_template, 
                    'examples/restaurant/hit_ids.txt','hit_templates/restaurant.html',
                    'hit_properties/restaurant.json', 'examples/restaurant/results.txt',
                    n_workers=2, sandbox=True, dirty_is_no=False)
    end_time = time.time()
    pickle.dump(estimates, open('examples/restaurant/t_walk.p','wb'))
    print('It took %s.'%(end_time - start_time))

def process_estimates():
    import pickle
    estimates = pickle.load(open('examples/restaurant/t_walk.p','rb'))

    w_range = np.arange(10,500,10)
    X, Y, GT = list(), list(), list()
    for w in w_range:
        X.append(w)
        Y.append(estimates[w])
        GT.append(12)

    with open('examples/restaurant/temp.txt','w') as f:
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
    fig.savefig('examples/restaurant/t_walk_fig.png', bbox_inches='tight')
    '''


if __name__ == '__main__':
    #test_exp_restaurant_dataset()
    test_triangular_walk()
    #process_estimates()
