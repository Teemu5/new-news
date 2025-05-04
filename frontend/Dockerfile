FROM nginx:alpine

#RUN rm -rf /usr/share/nginx/html/*

COPY index.html /usr/share/nginx/html/index.html
COPY favicon.ico /usr/share/nginx/html/favicon.ico

RUN touch /usr/share/nginx/html/wpad.dat

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]