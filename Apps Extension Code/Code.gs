function include(File) {
  return HtmlService.createHtmlOutputFromFile(File).getContent();
};

function onOpen() {
  DocumentApp.getUi()
      .createMenu('EssAI')
      .addItem("Launch", 'showDialog')
      /*
      .addSubMenu(DocumentApp.getUi().createMenu('Sentence')
        .addItem('Austen, Jane', 'showDialog_Austen')
        .addItem('Twain, Mark','showDialog_Twain')
      )
      .addSubMenu(DocumentApp.getUi().createMenu('Stylometrics')
        .addItem('Austen, Jane', 'showDialog2_Austen')
        .addItem('Twain, Mark','showDialog2_Twain')
      )*/
      .addToUi();
}

function showDialog() {
  var author = "Austen, Jane";
  var html_file = "main";
  var names = author.split(",");
  var last_name = names[0];
  var first_name = names[1];
  var userText = DocumentApp.getActiveDocument().getBody().getText();

  var tmp = HtmlService.createTemplateFromFile(html_file);
  tmp.docText = userText;
  tmp.author = last_name;

  var html = tmp.evaluate();
  var title = 'EssAI:  ' + first_name + " " + last_name;
  html.setWidth(1000)
      .setHeight(1500);
  
  DocumentApp.getUi().showModalDialog(html, " ");
}

function showDialog2(author) {
  var html_file = "Kyle2";
  var names = author.split(",");
  var last_name = names[0];
  var first_name = names[1];
  var userText = DocumentApp.getActiveDocument().getBody().getText();

  var tmp = HtmlService.createTemplateFromFile(html_file);
  tmp.docText = userText;
  tmp.author = last_name;

  var html = tmp.evaluate();
  var title = 'EssAI:  ' + first_name + " " + last_name;
  html.setWidth(1000)
      .setHeight(1500);
  
  DocumentApp.getUi().showModalDialog(html, title);
}

function showDialog_Austen() {showDialog('Austen, Jane');}
function showDialog_Twain() {showDialog('Twain, Mark');}
function showDialog2_Austen() {showDialog2('Austen, Jane');}
function showDialog2_Twain() {showDialog2('Twain, Mark');}
