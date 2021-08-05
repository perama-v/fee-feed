# fee-feed
Minimal Ethereum fee data viewer for the terminal, contained in a single python script. Connects to your node and displays some
metrics in real-time.

Boxplots of fees for the past few hundred blocks to give context
to the fee market.

![img](/img/mainnet_9.png)

See how much ether is burned recently.

![img](/img/goerli_2.png)

See the fee parameters for the last block in detail.

![img](/img/mainnet_1.png)

### Installation

Optionally create an environment with python 3.x

    python3 -m venv venv
    source venv/bin/activate

Install dependencies

    pip install requests python-dotenv


### Node configuration

Create a file called '.env' and populated it with the url for your 
node as follows:

    RPC_URL='http://localhost:8545'

### Operation

(optional) activate environment:

    source venv/bin/activate

Run

    python fee-feed.py

On startup, details of transactions in the first block will be displayed.
After the display is started, the receipts of the last 200 blocks
are fetched, which can take a minute for a local node.

Control with the keyboard:

- number to change mode
- q to quit.

### Stats

See boxplots with outliers removed

Mainnet:
![img](/img/mainnet_14.png)

Standard deviation of the priority fees.

See what multiple of the base fee the average max fee is.

Mainnet:
![img](/img/mainnet_4.png)

Goerli:
![img](/img/goerli_1.png)

---

What is the gas utilisation percentage?

How does the minimum fee change as block size changes, ignoring MEV transactions?

How many transactions are of the new type vs legacy?

Mainnet:
![img](/img/mainnet_3.png)

Goerli:
![img](/img/goerli_3.png)

---


Dynamically adjusts if you resize the window to handle large or small
monitors.

![img](/img/mainnet_5.png)

Pop it to the side to keep an eye on things.

![img](/img/mainnet_13.png)

Keep it out of the way.

![img](/img/mainnet_12.png)


## Networks

Functions with any testnet without configuration, tested on:

- Mainnet pre-London
- Goerli post-London.

## Upgrades

You can add new modes pretty easily:

1. Add a mode the configuration section at the
top of the file, replicating what is already there, and changing the
names and button.
2. Add a function to populate your newly named data. Search for
"# Add new data functions here."

## Issues

The boxplots are calculated on quartiles for maxPriorityFeePerGas,
weighted by gas - but the gas value is gas limit, rather than
gas used for that transaction. The prices are correct, but the
distribution might be off compared with feeHistory API results.

## Disclaimer

This is provided as-is for education purposes and should not be used as critical infrastructure.

## Feedback

PRs welcome. If you like it, or have ideas and feedback reach out to me on twitter @eth_worm
