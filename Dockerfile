FROM node:lts-alpine

WORKDIR /usr/src/app

COPY package.json yarn.lock ./

RUN yarn cache clean
RUN yarn install --production=false

COPY . .

RUN yarn build

EXPOSE 3000

CMD ["yarn", "start"]