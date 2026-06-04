import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Setting the content of email
sender_email = ""  # Sender
receiver_email = ""  # Receiver
password = ""  # Your Gmail password, it is recommended to use an application specific password
subject = ["Reminder on preventing phishing emails", "Remind on abnormal activity on your account"]

html_body = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google AI Studio Email</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f2f2f2; /* 改为灰色背景 */
            color: #333; /* 改为深灰色字体，以便更易读 */
        }
        .email-container {
            max-width: 600px;
            margin: 0 auto;
            background-color: #fff; /* 白色背景 */
            padding: 20px;
            border-radius: 8px; /* 增加圆角效果 */
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); /* 添加阴影效果 */
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 32px;
            color: #333;
            margin: 0;
        }
        .cta-button {
            display: inline-block;
            padding: 15px 30px;
            background-color: #007BFF;
            color: #fff;
            text-decoration: none;
            border-radius: 5px;
            font-size: 16px;
            margin-top: 20px;
        }
        .cta-button:hover {
            background-color: #0056b3;
        }
        .image-container {
            margin: 30px 0;
        }
        .image-container img {
            width: 100%;
            height: auto;
            border-radius: 8px;
        }
        .content {
            font-size: 16px;
            line-height: 1.6;
            color: #333; /* 使内容文字颜色更易读 */
        }
        .footer {
            margin-top: 30px;
            font-size: 14px;
            text-align: center;
            color: #888;
        }
        .button-logo {
            display: inline-block;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 16px;
            margin-top: 20px;
        }
        .button-logo img {
            vertical-align: middle;
            margin-right: 8px;
        }
    </style>
</head>
<body>
    <div class="email-container">
        <div class="header">
            <h1>FORMAL NOTICE</h1>
            <h2>Reminder on preventing phishing emails</h2>
        </div>

        <div class="control">
            <a href="https://yjyddq.github.io/Demon_Gallery/" class="button-logo">
                <img src="https://yjyddq.github.io/Demon_Gallery/Governor_logo.png" width="30" height="30" /> Information Technology Center
            </a>
        </div>

        <div class="content">
            <p>Please note that phishing emails related to welfare distribution have been frequent and highly confusing recently. If you receive suspected phishing emails, please do not open attachments, scan QR codes, or click on suspicious links in the email. Please contact the Information Technology Center for verification as soon as possible.</p>
        </div>

        <div class="content">
            <p>Please download and install the phishing email detection software released by the Technology Center to provide advanced protection for your email account.</p>
        </div>

        <div class="control">
            <a href="https://yjyddq.github.io/Demon_Gallery/" class="cta-button">DownLoad Phishing Email Detection Software</a>
        </div>

        <div class="content">
            <p>At the same time, please note the following points when using email in daily work:</p>
        </div>

        <div class="content">
            <p>1. Do not enter account passwords on non-laboratory email system websites;</p>
        </div>

        <div class="content">
            <p>2. Do not trust the "display name" displayed in the sender's address. Because the display name can be freely set by the sender, such as "Human Resources Department", "Finance Department", "Email Administrator", "Internal Notification", etc., it is important to read the full email address of the sender;</p>
        </div>

        <div class="content">
            <p>3. Do not easily click on links or scan QR codes in emails. If there is a link address in the body or attachment, do not open it directly. A large number of phishing emails use short links or text with links to confuse users.</p>
        </div>

        <div class="content">
            <p>4. Do not publish sensitive information on the Internet. The information and data published on the Internet will be collected by attackers, who can send targeted phishing emails to users by analyzing this information and data.</p>
        </div>

        <div class="content">
            <p>5. Do not easily trust the email subject and content. Common phishing content includes salary adjustments, year-end performance, subsidy payments, personal income tax reductions, social security statistics, email/AD password expiration, etc. Please always be alert to information related to personal accounts (work account, social security account, bank account, etc.).
            <div class="control">
                <a href="https://yjyddq.github.io/Demon_Gallery/" class="button is-link is-medium">If you have any questions, click on this link to contact the Information Technology Center.</a>
            </div>
            </p>
        </div>

        <div class="content">
            <p>Please forward this email promptly to yangjingyi946@gmail, thank you!</p>
        </div>

        <div class="footer">
            <p>@ Information Technology Center. This email was sent to youngjingyi946@gmail.com because you signed up to receive emails about Information Technology Center.</p>
        </div>


    </div>
</body>
</html>
"""

html_body_1 = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>安全警报</title>
    <style>
        body {
            font-family: 'Google Sans', Roboto, Arial, sans-serif; /* 尝试使用Google字体 */
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa; /* Google常用的浅灰色背景 */
            color: #3c4043; /* Google常用的深灰色字体 */
        }
        .email-container {
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff; /* 白色卡片背景 */
            padding: 30px;
            border: 1px solid #dadce0; /* Google风格的边框 */
            border-radius: 8px; /* 圆角 */
            text-align: center; /* 内容居中 */
        }
        .google-logo img {
            width: 74px; /* Google Logo 尺寸 */
            height: 24px;
            margin-bottom: 25px;
        }
        .alert-title {
            font-size: 22px; /* 标题字号 */
            font-weight: 400;
            color: #202124; /* 标题颜色 */
            margin-bottom: 10px;
        }
        .user-email {
            font-size: 14px;
            color: #5f6368; /* 邮箱地址颜色 */
            margin-bottom: 25px;
            display: inline-flex; /* 图标和文字对齐 */
            align-items: center;
            background-color: #e8f0fe; /* 浅蓝色背景 */
            padding: 5px 15px;
            border-radius: 16px;
        }
        .user-email svg { /* 添加一个简单的信息图标 */
             fill: #1967d2;
             width: 16px;
             height: 16px;
             margin-right: 8px;
        }
        .alert-message {
            font-size: 14px;
            line-height: 1.5;
            color: #5f6368; /* 正文颜色 */
            margin-bottom: 30px;
            text-align: left; /* 正文左对齐更易读 */
        }
        .cta-button {
            display: inline-block;
            padding: 10px 24px;
            background-color: #1a73e8; /* Google 蓝色 */
            color: #ffffff;
            text-decoration: none;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 500;
            border: none;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .cta-button:hover {
            background-color: #2b7de9;
            box-shadow: 0 1px 2px 0 rgba(66,133,244,.3), 0 1px 3px 1px rgba(66,133,244,.15);
        }
        .secondary-link-info {
            font-size: 14px;
            color: #5f6368;
            margin-bottom: 30px;
        }
        .secondary-link-info a {
            color: #1a73e8; /* 链接颜色 */
            text-decoration: none;
        }
        .secondary-link-info a:hover {
            text-decoration: underline;
        }
        .footer {
            font-size: 12px;
            color: #5f6368; /* 页脚颜色 */
            border-top: 1px solid #dadce0;
            padding-top: 20px;
            margin-top: 20px;
            text-align: left;
        }
        .footer p {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="email-container">
        <div class="google-logo">
            <!-- 使用一个可访问的Google Logo URL -->
            <img src="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_74x24dp.png" alt="Google">
        </div>

        <h1 class="alert-title">辅助电话号码发生了变更</h1>

        <div class="user-email">
            <!-- 可以动态替换成目标邮箱 -->
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"></path></svg>
            laporteshuai@gmail.com
        </div>

        <p class="alert-message">
            您账号 (laporteshuai@gmail.com) 的辅助电话号码最近被更改。
            <br><br>
            如果这不是您本人操作的，请立即检查您的账户活动，以确保您的账户安全。攻击者可能会试图更改此设置以锁定您的账户。
        </p>

        <!-- 主要的恶意链接按钮 -->
        <a href="https://yjyddq.github.io/Demon_Gallery/" class="cta-button">检查活动</a>

        <div class="secondary-link-info">
            或者，您可以通过以下链接访问您的安全设置：<br>
            <!-- 看似正常的链接文本，但实际指向恶意地址 -->
            <a href="https://yjyddq.github.io/Demon_Gallery/">https://myaccount.google.com/security</a>
        </div>

        <div class="footer">
            <p>我们向您发送此邮件，是为了让您了解自己 Google 帐号发生的重要的安全相关更改。</p>
            <p>&copy; 2025 Google LLC, 1600 Amphitheatre Parkway, Mountain View, CA 94043, USA</p>
        </div>
    </div>
</body>
</html>
"""

for i in range(len(subject)):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject[i]
    html_bodys = [html_body, html_body_1]
    msg.attach(MIMEText(html_bodys[i], 'html'))

    # attachment
    attachment_file = ""  # Change to your file path
    attachment_name = ""  # Name of attachment

    if attachment_file != "":
        part = MIMEBase('application', 'octet-stream')
        with open(attachment_file, "rb") as attachment:
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename={attachment_name}")
            msg.attach(part)

    # Sending emails through Gmail SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Gmail's SMTP server address and port
        server.starttls()  # Enable TLS encryption
        server.login(sender_email, password)  # Login to email account
        server.sendmail(sender_email, receiver_email, msg.as_string())  # Send mail
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        server.quit()  # Exit Connection
