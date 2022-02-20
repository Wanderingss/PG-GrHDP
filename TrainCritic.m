% TrainCritic.m
function [V, theta1, W, theta2, accuracy] = TrainCritic(data, Minaccuracy, Maxepochs, showWindow)
	shape = size(data);
	input = data(:,1:shape(2)-1);
	input = input';
	loss = data(:,shape(2));
	loss = loss';
	net_critic = feedforwardnet(20, 'trainlm');
	net_critic.trainParam.showWindow = showWindow;
	net_critic.biasConnect = [1;1];
	net_critic.input.processFcns = {};
	net_critic.output.processFcns ={};
	net_critic.trainparam.goal = Minaccuracy ;
	net_critic.trainParam.epochs = Maxepochs;       
	net_critic.trainParam.lr = 0.005;
	net_critic.layers{1}.transferFcn = 'logsig';
	net_critic = train(net_critic,input,loss);
	V = net_critic.iw{1,1};
  theta1 = net_critic.b{1};
	W = net_critic.lw{2,1};
	output = net_critic(input);
  theta2 = net_critic.b{2};
	accuracy = perform(net_critic,loss, output);



